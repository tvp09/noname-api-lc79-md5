
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const express = require('express');

// ==========================================
// CONFIGURATION
// ==========================================

const PORT = process.env.PORT || 3000;
const API_ID = process.env.API_ID || 'noname09';

const API_URL = process.env.API_URL ||
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?at=6256356b3f81d9bd5e0950ecd13c7df0&cl=R&cp=R&pf=web';

const FILE = path.join(
  process.env.DATA_DIR || __dirname,
  'lichsulc79md5.json'
);

const MAX = 200;
const MIN_WINDOW = 15;
const WINDOW = 200;

const WINDOWS = [
  15, 30, 60, 80, 100, 150, 200
];

const INTERVAL = 5000;
const VERSION = 'six-points-v8';

const app = express();

app.disable('x-powered-by');
app.set('json spaces', 2);

app.use((req, res, next) => {
  res.set('Cache-Control', 'no-store');
  res.set('Access-Control-Allow-Origin', '*');
  next();
});

// ==========================================
// UTILITIES
// ==========================================

const vnTime = () =>
  new Date(Date.now() + 25200000)
    .toISOString()
    .replace('Z', '+07:00');

const name = s =>
  s === 't' ? 'tai' : 'xiu';

const pct = p =>
  p.toFixed(2) + '%';

const scorePct = p =>
  Math.round(p) + '%';

const sequence = h =>
  h.map(r => r.result).join('');

const clip = p =>
  Math.max(0.001, Math.min(0.999, p));

const beta = (t, n, a = 3) =>
  (t + a) / (n + 2 * a);

const countT = s =>
  [...s].filter(c => c === 't').length;

// ==========================================
// DATABASE ENGINE
// ==========================================

const empty = () => ({
  history: [],
  sequence: '',
  predictions: {},
  statistics: []
});

function load() {
  if (!fs.existsSync(FILE)) {
    return empty();
  }

  const d = {
    ...empty(),
    ...JSON.parse(
      fs.readFileSync(FILE, 'utf8')
    )
  };

  if (
    !Array.isArray(d.history) ||
    !Array.isArray(d.statistics) ||
    !d.predictions ||
    typeof d.predictions !== 'object'
  ) {
    throw new Error('Invalid history file');
  }

  d.history = d.history.slice(-MAX);
  d.sequence = sequence(d.history);

  return d;
}

let db = load();

// ==========================================
// FIFO STORAGE
// MIN 15 - MAX 200
// ==========================================

function commit(next) {
  next.history =
    next.history.slice(-MAX);

  next.sequence =
    sequence(next.history);

  next.statistics =
    next.statistics.slice(-MAX);

  fs.mkdirSync(
    path.dirname(FILE),
    { recursive: true }
  );

  fs.writeFileSync(
    FILE + '.tmp',
    JSON.stringify(next, null, 2),
    'utf8'
  );

  fs.renameSync(FILE + '.tmp', FILE);

  db = next;
}

function recent(h, start) {
  const rows = h
    .filter(r => r.id >= start)
    .slice(-WINDOW);

  let i = rows.length - 1;

  while (
    i > 0 &&
    rows[i].id === rows[i - 1].id + 1
  ) {
    i--;
  }

  return sequence(
    rows.slice(Math.max(0, i))
  );
}

// ==========================================
// ALGORITHM 01
// BAYESIAN FREQUENCY
// ==========================================

function bayesian(s) {
  return beta(
    countT(s),
    s.length,
    4
  );
}

// ==========================================
// ALGORITHM 02
// SHORT TERM EWMA
// ==========================================

function ewma(s) {
  let t = 0;
  let n = 0;

  for (let i = 0; i < s.length; i++) {
    const age = s.length - 1 - i;

    const w = 2 ** (-age / 6);

    t += w * Number(s[i] === 't');
    n += w;
  }

  return beta(t, n, 3);
}

// ==========================================
// ALGORITHM 03
// SYMMETRIC MARKOV CHAIN
// ==========================================

// Hoc kha nang giu cau va dao cau.
// Khong mac dinh uu tien TAI hay XIU.

function markov(s) {
  if (s.length < 2) {
    return 0.5;
  }

  let same = 0;
  let n = 0;

  for (let i = 1; i < s.length; i++) {
    const w = 2 ** (
      -(s.length - 1 - i) / 24
    );

    n += w;

    same += w * Number(
      s[i] === s[i - 1]
    );
  }

  const p = beta(same, n, 3);

  return s.at(-1) === 't'
    ? p
    : 1 - p;
}

// ==========================================
// ALGORITHM 04
// VARIABLE ORDER MARKOV
// MIRRORED CONTEXT MATCHING
// ==========================================

// Nhan dien mau cau goc
// va mau cau dao nguoc TAI/XIU.
//
// TTX tuong duong XXT
// TTXX tuong duong XXTT

function variableMarkov(s) {
  if (s.length < 2) {
    return 0.5;
  }

  const len = s.length;
  const last = s.at(-1);

  let p = 0.5;

  for (
    let k = 1;
    k <= Math.min(7, len - 1);
    k++
  ) {
    let same = 0;
    let n = 0;

    for (let j = k; j < len; j++) {
      const mirror =
        s[j - 1] !== last;

      let match = true;

      for (let d = 1; d <= k; d++) {
        if (
          (s[j - d] !== s[len - d])
          !== mirror
        ) {
          match = false;
          break;
        }
      }

      if (!match) {
        continue;
      }

      const w = 2 ** (
        -(len - 1 - j) / 30
      );

      n += w;

      same += w * Number(
        s[j] === s[j - 1]
      );
    }

    const prior = 5 + k;

    p = (same + prior * p) /
        (n + prior);
  }

  return last === 't'
    ? p
    : 1 - p;
}

// ==========================================
// ALGORITHM 05
// ADAPTIVE RUN LENGTH
// ==========================================

// Phan tich do dai cau.
// Hoc kha nang tiep tuc va chuyen cau.
// Su dung ca cac do dai gan tuong dong.

function runLength(s) {
  if (s.length < 2) {
    return 0.5;
  }

  const len = s.length;

  let target = 1;

  for (
    let i = len - 2;
    i >= 0 && s[i] === s.at(-1);
    i--
  ) {
    target++;
  }

  let run = 1;
  let same = 0;
  let n = 0;
  let known = false;

  for (let i = 1; i < len - 1; i++) {
    if (s[i] !== s[i - 1]) {
      run = 1;
      known = true;
    } else {
      run++;
    }

    if (!known) {
      continue;
    }

    const distance = Math.abs(
      Math.min(run, 8) -
      Math.min(target, 8)
    );

    const w =
      2 ** (-(len - 2 - i) / 24) *
      2 ** (-2 * distance);

    n += w;

    same += w * Number(
      s[i + 1] === s[i]
    );
  }

  const p = beta(same, n, 3);

  return s.at(-1) === 't'
    ? p
    : 1 - p;
}

// ==========================================
// ALGORITHM 06
// ADAPTIVE PATTERN RECOGNITION
// ==========================================

// Phan tich cac mau lap co chu ky:
//
// Cau 1-1
// Cau 2-1
// Cau 1-2
// Cau 2-2
// Cau 3-1
// Cau 1-3
// Cau 3-2
// Cau 2-3
// Cau 3-3
// Cau 4-4
//
// Danh gia chu ky tu 2 den 8.
// Khong bat buoc mau lap trung khop 100%.

function patternRecognition(s) {
  const len = s.length;

  let weighted = 0;
  let total = 1;

  for (
    let period = 2;
    period <= 8;
    period++
  ) {
    if (len < 2 * period + 1) {
      continue;
    }

    // Danh gia muc do khop
    // cua chu ky gan nhat.

    const checks = Math.min(
      2 * period,
      len - period
    );

    let local = 0;

    for (
      let j = len - checks;
      j < len;
      j++
    ) {
      local += Number(
        s[j] === s[j - period]
      );
    }

    const localFit = Math.max(
      0,
      2 * beta(local, checks, 1) - 1
    );

    if (!localFit) {
      continue;
    }

    // Kiem tra chu ky tren lich su.

    let hits = 0;
    let n = 0;

    for (
      let j = period;
      j < len;
      j++
    ) {
      const w = 2 ** (
        -(len - 1 - j) / 30
      );

      n += w;

      hits += w * Number(
        s[j] === s[j - period]
      );
    }

    const p = beta(hits, n, 3);

    const support = Math.max(
      0,
      2 * p - 1
    );

    // Chi tang diem khi co ca
    // tin hieu hien tai va ho tro lich su.

    const weight =
      localFit *
      support *
      n / (n + 6);

    const signal =
      s[len - period] === 't'
        ? support
        : -support;

    weighted += weight * signal;
    total += weight;
  }

  return clip(
    0.5 + weighted / (2 * total)
  );
}

// ==========================================
// EXACTLY SIX MODELS
// ==========================================

const models = [
  bayesian,
  ewma,
  markov,
  variableMarkov,
  runLength,
  patternRecognition
];

// ==========================================
// ADAPTIVE MULTI WINDOW ENGINE V8
// ==========================================

function predict(input) {
  const s = input.slice(-WINDOW);

  if (/[^tx]/.test(s)) {
    throw new Error(
      'Invalid T/X sequence'
    );
  }

  if (s.length < MIN_WINDOW) {
    return null;
  }

  // Tu dong mo rong so phien phan tich.

  const windows = WINDOWS.filter(
    w => w <= s.length
  );

  if (!windows.includes(s.length)) {
    windows.push(s.length);
  }

  const states = models.map(model => ({
    model,

    windows: windows.map(size => ({
      size,
      fast: 0,
      slow: 0,
      nf: 0,
      ns: 0
    }))
  }));

  // ========================================
  // DUAL SPEED PERFORMANCE LEARNING
  // ========================================

  // Danh gia hieu suat ngan han
  // va dai han cua tung mo hinh.

  const fastDecay = 2 ** (-1 / 12);
  const slowDecay = 2 ** (-1 / 40);

  // ========================================
  // WALK FORWARD VALIDATION
  // ========================================

  // Khong nhin truoc ket qua.
  // So sanh mo hinh voi moc 50/50.

  for (let i = 8; i < s.length; i++) {
    const actual = s[i] === 't';

    for (const state of states) {

      // Cache tranh tinh trung cac cua so.

      const cache = new Map();

      for (const item of state.windows) {
        const size = Math.min(
          i,
          item.size
        );

        if (!cache.has(size)) {
          const prefix = s.slice(
            i - size,
            i
          );

          cache.set(
            size,
            clip(state.model(prefix))
          );
        }

        const p = cache.get(size);

        const gain = Math.log(
          2 * (actual ? p : 1 - p)
        );

        item.fast =
          fastDecay * item.fast + gain;

        item.slow =
          slowDecay * item.slow + gain;

        item.nf =
          fastDecay * item.nf + 1;

        item.ns =
          slowDecay * item.ns + 1;
      }
    }
  }

  // ========================================
  // ADAPTIVE WINDOW EVALUATION
  // ========================================

  const evaluated = states.map(state => {
    const candidates = state.windows.map(
      item => {
        const p = clip(
          state.model(
            s.slice(-item.size)
          )
        );

        // 65% hieu suat gan day.
        // 35% hieu suat dai han.

        const quality =
          0.65 *
          item.fast / (item.nf + 8) +

          0.35 *
          item.slow / (item.ns + 12);

        const weight =
          Math.exp(12 * quality);

        return {
          size: item.size,
          p,
          quality,
          weight
        };
      }
    );

    const total = candidates.reduce(
      (a, c) => a + c.weight,
      0
    );

    const probability =
      candidates.reduce(
        (a, c) =>
          a + c.p * c.weight / total,
        0
      );

    const quality =
      candidates.reduce(
        (a, c) =>
          a + c.quality *
          c.weight / total,
        0
      );

    return {
      model: state.model.name,

      probability,

      quality,

      windows: candidates.map(c => ({
        size: c.size,

        probabilityTai: c.p,

        quality: c.quality,

        weight: c.weight / total
      }))
    };
  });

  // ========================================
  // ADAPTIVE MODEL WEIGHTING
  // ========================================

  const raw = evaluated.map(
    c => Math.exp(12 * c.quality)
  );

  const total = raw.reduce(
    (a, w) => a + w,
    0
  );

  let taiPoints = 0;
  let xiuPoints = 0;

  // ========================================
  // SYMMETRIC MODEL POINT SCORING
  // ========================================

  const modelPoints = evaluated.map(
    (c, i) => {
      const weight = raw[i] / total;

      const signed =
        100 *
        weight *
        (2 * c.probability - 1);

      const tai = Math.max(
        0,
        signed
      );

      const xiu = Math.max(
        0,
        -signed
      );

      taiPoints += tai;
      xiuPoints += xiu;

      return {
        model: c.model,

        probabilityTai: c.probability,

        weight,

        quality: c.quality,

        tai,

        xiu,

        windows: c.windows
      };
    }
  );

  // ========================================
  // FINAL DECISION
  // ========================================

  const net = taiPoints - xiuPoints;

  const side =
    Math.abs(net) > 1e-10

      ? (net > 0 ? 't' : 'x')

      : (s.at(-1) === 't' ? 'x' : 't');

  // ========================================
  // SIGNAL STRENGTH
  // ========================================

  const strength = Math.min(
    1,
    Math.abs(net) / 100
  );

  // Muc do dong thuan cua cac mo hinh.

  const agreement =
    Math.abs(net) /

    Math.max(
      1e-10,
      taiPoints + xiuPoints
    );

  // ========================================
  // NONLINEAR SIGNAL SCORE 51-99
  // ========================================

  // Diem tin hieu, khong phai
  // xac suat thang thuc te.
  //
  // Dat 99 khi tin hieu du manh
  // va cac mo hinh dong thuan.
  //
  // Khong random.
  // Khong ep tang giam theo phien.

  const smooth = Math.min(
    1,

    Math.log1p(5 * strength) /
    Math.log1p(5 * 0.75)
  );

  const confidence = Math.max(
    51,

    Math.min(
      99,

      Math.round(
        51 +
        48 *
        smooth *
        (0.65 + 0.35 * agreement)
      )
    )
  );

  // ========================================
  // STATISTICAL RELIABILITY
  // ========================================

  const quality = modelPoints.reduce(
    (a, c) =>
      a + c.quality * c.weight,
    0
  );

  const positive = Math.max(
    0,
    quality
  );

  const reliability =
    positive / (positive + 0.05);

  return {
    side,

    confidence,

    scoreType: 'model_signal_score',

    probabilityTai: clip(
      0.5 + net * reliability / 200
    ),

    taiPoints,

    xiuPoints,

    modelPoints,

    reliability,

    analyzedSessions: s.length,

    analysisWindows: windows
  };
}

// ==========================================
// SOURCE API PARSER
// ==========================================

function parse(payload) {
  if (!Array.isArray(payload.list)) {
    throw new Error(
      'API schema: missing list'
    );
  }

  const map = new Map();

  for (const r of payload.list) {
    const id = Number(r.id);

    const result =
      r.resultTruyenThong;

    if (
      !Number.isSafeInteger(id) ||
      id <= 0 ||
      !['TAI', 'XIU'].includes(result)
    ) {
      continue;
    }

    const dices = (
      Array.isArray(r.dices)

        ? r.dices

        : String(r.dices ?? '')
            .split(/[-,\s]+/)
    ).map(Number);

    if (
      dices.length !== 3 ||
      !dices.every(d =>
        Number.isInteger(d) &&
        d >= 1 &&
        d <= 6
      )
    ) {
      continue;
    }

    const point = Number(r.point);

    if (
      point !== dices.reduce(
        (a, b) => a + b,
        0
      )
    ) {
      continue;
    }

    map.set(id, {
      id,

      result:
        result === 'TAI' ? 't' : 'x',

      dices,

      point
    });
  }

  return [...map.values()].sort(
    (a, b) => a.id - b.id
  );
}

// ==========================================
// PROCESS NEW SESSIONS
// ==========================================

function apply(sessions) {
  if (!sessions.length) {
    return false;
  }

  const last =
    db.history.at(-1)?.id || 0;

  if (sessions.at(-1).id < last) {
    throw new Error(
      'Source behind saved history'
    );
  }

  const migration =
    db.modelVersion !== VERSION ||
    !Number.isSafeInteger(db.modelStart);

  const incoming = sessions.filter(
    r => r.id > last
  );

  // Khong tinh lai neu chua co phien moi.

  if (!migration && !incoming.length) {
    return false;
  }

  const next = {
    ...db,

    history: [...db.history],

    predictions: {
      ...db.predictions
    },

    statistics: [...db.statistics]
  };

  // ========================================
  // MODEL VERSION MIGRATION
  // ========================================

  if (migration) {
    next.modelVersion = VERSION;

    next.modelStart =
      [
        'six-points-v2',
        'six-points-v3',
        'six-points-v4',
        'six-points-v5',
        'six-points-v6',
        'six-points-v7'
      ].includes(db.modelVersion) &&

      Number.isSafeInteger(db.modelStart)

        ? db.modelStart

        : sessions[0].id;

    const merged = new Map(
      next.history.map(
        r => [r.id, r]
      )
    );

    for (const r of sessions) {
      merged.set(r.id, r);
    }

    next.history = [...merged.values()]
      .sort((a, b) => a.id - b.id);

  } else {
    next.history.push(...incoming);
  }

  // ========================================
  // CHECK PREVIOUS PREDICTIONS
  // ========================================

  for (const r of incoming) {
    const old =
      next.predictions[String(r.id)];

    if (!old) {
      continue;
    }

    next.statistics.push({
      id: r.id,

      predicted: old.side,

      actual: r.result,

      confidence: old.confidence,

      scoreType:
        old.scoreType ||
        'legacy_estimate',

      correct:
        old.side === r.result
    });

    console.log(
      '[CHECK]',
      r.id,

      old.side === r.result
        ? 'DUNG'
        : 'SAI'
    );
  }

  // ========================================
  // FIFO STORAGE MAX 200
  // ========================================

  next.history =
    next.history.slice(-MAX);

  const latest =
    next.history.at(-1).id;

  const key =
    String(latest + 1);

  const analysis = recent(
    next.history,
    next.modelStart
  );

  // Du 15 phien moi bat dau du doan.
  // Giu nguyen du doan da cong bo.

  const prediction =
    analysis.length >= MIN_WINDOW

      ? (
          next.predictions[key] ||
          predict(analysis)
        )

      : null;

  next.predictions = prediction

    ? { [key]: prediction }

    : {};

  commit(next);

  // ========================================
  // CONSOLE LOGS
  // ========================================

  if (prediction) {
    console.log(
      '[PHIEN]',
      latest,

      '[KET QUA]',
      name(next.history.at(-1).result)
    );

    console.log(
      '[DU DOAN]',
      name(prediction.side),

      '[DIEM TIN HIEU]',
      scorePct(prediction.confidence)
    );

    console.log(
      '[DIEM TAI]',
      prediction.taiPoints?.toFixed(4),

      '[DIEM XIU]',
      prediction.xiuPoints?.toFixed(4)
    );

    console.log(
      '[PHAN TICH]',
      analysis.length + '/200',

      '[WINDOWS]',
      prediction.analysisWindows
    );

  } else {
    console.log(
      '[WAITING]',
      analysis.length + '/' + MIN_WINDOW
    );
  }

  return true;
}

// ==========================================
// AUTOMATIC API POLLING
// ==========================================

let lastSuccess = 0;
let lastError = false;

async function poll() {
  try {
    const res = await fetch(API_URL, {
      method: 'GET',

      headers: {
        Accept: 'application/json'
      },

      signal:
        AbortSignal.timeout(4500)
    });

    if (!res.ok) {
      throw new Error(
        'HTTP ' + res.status
      );
    }

    const sessions = parse(
      await res.json()
    );

    if (!sessions.length) {
      throw new Error(
        'No valid sessions'
      );
    }

    apply(sessions);

    lastSuccess = Date.now();
    lastError = false;

  } catch (e) {
    lastError = true;

    console.error(
      '[POLL ERROR]',
      e.message
    );
  }
}

// ==========================================
// MAIN API OUTPUT
// ==========================================

function output() {
  const r = db.history.at(-1);

  const p = r &&
    db.predictions[String(r.id + 1)];

  if (
    !r ||
    !p ||
    db.modelVersion !== VERSION
  ) {
    return null;
  }

  return {
    id: API_ID,

    phienhientai: r.id,

    ketqua: name(r.result),

    xucxac: r.dices,

    tong: r.point,

    dudoanphien: r.id + 1,

    dudoan: name(p.side),

    dotincay: scorePct(p.confidence),

    uptime: vnTime()
  };
}

// ==========================================
// BEAUTIFUL JSON FORMAT
// ==========================================

function pretty(data) {
  return '{\n' +

    Object.entries(data).map(
      ([k, v]) =>

        '  ' +
        JSON.stringify(k) +
        ': ' +

        (
          Array.isArray(v)

            ? '[' +
              v.map(x =>
                JSON.stringify(x)
              ).join(', ') +
              ']'

            : JSON.stringify(v)
        )

    ).join(',\n') +

    '\n}';
}

// ==========================================
// MAIN API ENDPOINT
// ==========================================

app.get(['/', '/api'], (req, res) => {
  const data = output();

  if (!data) {
    return res.status(503).json({
      error: 'Dang cho du lieu API'
    });
  }

  const p =
    db.predictions[
      String(data.dudoanphien)
    ];

  res.set(
    'X-Dotincay-Meaning',

    p.scoreType === 'model_signal_score'

      ? 'Model signal score, not win probability'

      : 'Legacy probability estimate, not calibrated'
  );

  return res
    .type('application/json')
    .send(pretty(data));
});

// ==========================================
// HISTORY ENDPOINT
// ==========================================

app.get('/lichsu', (req, res) => {
  return res.json({
    id: API_ID,

    sophien: db.history.length,

    max: MAX,

    sophienphantich:
      db.modelVersion === VERSION

        ? recent(
            db.history,
            db.modelStart
          ).length

        : 0,

    chuoi: db.sequence,

    history: db.history
  });
});

// ==========================================
// STATISTICS ENDPOINT
// ==========================================

app.get('/thongke', (req, res) => {
  const rows =
    db.statistics.slice(-10);

  const wins = rows.filter(
    r => r.correct
  ).length;

  return res.json({
    id: API_ID,

    sophien: rows.length,

    dung: wins,

    sai: rows.length - wins,

    tyle: rows.length

      ? pct(
          100 * wins / rows.length
        )

      : null,

    chuoi: rows.map(
      r => r.correct ? 'W' : 'L'
    ).join(''),

    chitiet: rows.map(r => ({
      phien: r.id,

      dudoan: name(r.predicted),

      ketqua: name(r.actual),

      trangthai:
        r.correct ? 'DUNG' : 'SAI',

      dotincay:
        scorePct(r.confidence),

      loaidotincay:
        r.scoreType ||
        'legacy_estimate'
    }))
  });
});

// ==========================================
// HEALTH CHECK
// ==========================================

app.get('/health', (req, res) => {
  return res.json({
    status:
      lastError ||
      (
        lastSuccess &&
        Date.now() - lastSuccess > 15000
      )

        ? 'degraded'

        : lastSuccess

          ? 'online'

          : 'waiting',

    lastSuccess: lastSuccess

      ? new Date(
          lastSuccess + 25200000
        ).toISOString()
         .replace('Z', '+07:00')

      : null,

    history: db.history.length,

    uptime: process.uptime()
  });
});

// ==========================================
// AUTO LOOP EVERY 5 SECONDS
// ==========================================

async function loop() {
  const started = Date.now();

  await poll();

  setTimeout(
    loop,

    Math.max(
      0,
      INTERVAL - (Date.now() - started)
    )
  );
}

// ==========================================
// START SERVER
// ==========================================

if (require.main === module) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(
      'LC79 MD5 API ONLINE',
      VERSION,
      'PORT:',
      PORT
    );

    console.log(
      'MODELS:', models.length,
      'MIN:', MIN_WINDOW,
      'MAX:', WINDOW,
      'FIFO:', MAX,
      'RANDOM: DISABLED'
    );

    loop();
  });
}

module.exports = {
  predict,
  parse,
  pretty,
  apply,
  output
};
