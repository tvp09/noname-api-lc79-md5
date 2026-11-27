
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

// Bat dau du doan tu 15 phien.
// Tich luy toi da 200 phien.
// Phien 201 xoa phien cu nhat.

const MAX = 200;
const MIN_WINDOW = 15;
const WINDOW = 200;

const WINDOWS = [
  15,
  30,
  60,
  80,
  100,
  150,
  200
];

const INTERVAL = 5000;
const VERSION = 'six-points-v7';

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

  const data = {
    ...empty(),
    ...JSON.parse(
      fs.readFileSync(FILE, 'utf8')
    )
  };

  if (
    !Array.isArray(data.history) ||
    !Array.isArray(data.statistics) ||
    !data.predictions ||
    typeof data.predictions !== 'object'
  ) {
    throw new Error('Invalid history file');
  }

  data.history = data.history.slice(-MAX);
  data.sequence = sequence(data.history);

  return data;
}

let db = load();

// ==========================================
// FIFO STORAGE - MAX 200 SESSIONS
// ==========================================

function commit(next) {

  // Chi giu 200 phien moi nhat.
  // Tu phien 201 tu dong xoa phien cu.

  next.history = next.history.slice(-MAX);

  next.sequence = sequence(next.history);

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

// ==========================================
// RECENT DATA FILTER
// MINIMUM 15 - MAXIMUM 200 SESSIONS
// ==========================================

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
// BAYESIAN ESTIMATION
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
// EXPONENTIALLY WEIGHTED MOVING AVERAGE
// ==========================================

function ewma(s) {
  let t = 0;
  let n = 0;

  for (let i = 0; i < s.length; i++) {
    const age = s.length - 1 - i;

    const w = 2 ** (-age / 8);

    t += w * Number(s[i] === 't');
    n += w;
  }

  return beta(t, n, 3);
}

// ==========================================
// CONTEXT MATCHING CORE
// ==========================================

function contextCounts(s, k) {
  if (s.length <= k) {
    return [0, 0];
  }

  const suffix = s.slice(-k);

  let t = 0;
  let n = 0;

  for (let i = 0; i + k < s.length; i++) {
    if (s.slice(i, i + k) !== suffix) {
      continue;
    }

    const age =
      s.length - 1 - (i + k);

    const w = 2 ** (-age / 18);

    n += w;

    t += w * Number(
      s[i + k] === 't'
    );
  }

  return [t, n];
}

// ==========================================
// ALGORITHM 03
// ADAPTIVE MARKOV CHAIN
// ==========================================

function markov(s) {
  return beta(
    ...contextCounts(s, 1),
    3
  );
}

// ==========================================
// ALGORITHM 04
// VARIABLE ORDER MARKOV
// ==========================================

function variableMarkov(s) {
  let p = 0.5;

  const maxOrder = Math.min(
    6,
    s.length - 1
  );

  for (let k = 1; k <= maxOrder; k++) {
    const [t, n] = contextCounts(s, k);

    const prior = 5 + k;

    p = (t + prior * p) /
        (n + prior);
  }

  return p;
}

// ==========================================
// ALGORITHM 05
// ADAPTIVE RUN LENGTH ANALYSIS
// ==========================================

function runLength(s) {
  if (s.length < 2) {
    return 0.5;
  }

  const side = s.at(-1);

  let target = 1;

  for (
    let i = s.length - 2;
    i >= 0 && s[i] === side;
    i--
  ) {
    target++;
  }

  let t = 0;
  let n = 0;
  let run = 0;

  let knownStart = false;

  for (let i = 0; i < s.length - 1; i++) {
    if (
      i > 0 &&
      s[i] !== s[i - 1]
    ) {
      knownStart = true;
      run = 1;
    } else {
      run++;
    }

    if (
      !knownStart ||
      s[i] !== side
    ) {
      continue;
    }

    if (
      Math.min(run, 5) !==
      Math.min(target, 5)
    ) {
      continue;
    }

    const w = 2 ** (
      -(s.length - 2 - i) / 18
    );

    n += w;

    t += w * Number(
      s[i + 1] === 't'
    );
  }

  return beta(t, n, 3);
}

// ==========================================
// ALGORITHM 06
// ADAPTIVE PATTERN RECOGNITION
// ==========================================

// Nhan dien mau cau:
// Cau bet
// Cau dao 1-1
// Cau 2-1
// Cau 1-2
// Cau 2-2
// Cau 3-3
// Cau 3-1
// Chu ky lap tu 1 den 6

function patternRecognition(s) {
  const hits = new Map();

  const len = s.length;

  for (
    let period = 1;
    period <= 6;
    period++
  ) {
    if (len < 2 * period) {
      continue;
    }

    const block = s.slice(-period);

    let primitive = true;

    for (let d = 1; d < period; d++) {
      if (
        period % d === 0 &&
        block === block
          .slice(0, d)
          .repeat(period / d)
      ) {
        primitive = false;
        break;
      }
    }

    if (!primitive) {
      continue;
    }

    let span = period;

    while (
      span < len &&
      s[len - 1 - span] ===
      s[len - 1 - span + period]
    ) {
      span++;
    }

    if (span < 2 * period) {
      continue;
    }

    const high = Math.min(
      span,
      2 * period + 2,
      len - 1
    );

    for (
      let k = period + 1;
      k <= high;
      k++
    ) {
      const suffix = s.slice(-k);

      const quality =
        k / Math.max(
          2 * period + 2,
          k
        );

      for (let j = k; j < len; j++) {
        if (
          s.slice(j - k, j) !== suffix
        ) {
          continue;
        }

        const w =
          quality *
          2 ** (-(len - 1 - j) / 18);

        hits.set(
          j,
          Math.max(
            hits.get(j) || 0,
            w
          )
        );
      }
    }
  }

  let t = 0;
  let n = 0;

  for (const [j, w] of hits) {
    n += w;

    t += w * Number(
      s[j] === 't'
    );
  }

  return beta(t, n, 3);
}

// ==========================================
// EXACTLY SIX PREDICTION MODELS
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
// MULTI WINDOW ADAPTIVE SCORING V7
// ==========================================

function predict(input) {
  const s = input.slice(-WINDOW);

  if (/[^tx]/.test(s)) {
    throw new Error(
      'Invalid T/X sequence'
    );
  }

  // Bat dau du doan ngay khi du 15 phien.

  if (s.length < MIN_WINDOW) {
    return null;
  }

  // Tu dong mo rong cua so phan tich
  // theo so luong phien hien co.

  const windows = WINDOWS.filter(
    w => w <= s.length
  );

  if (!windows.includes(s.length)) {
    windows.push(s.length);
  }

  const forget = 2 ** (-1 / 30);

  const states = models.map(model => ({
    model,

    windows: windows.map(size => ({
      size,
      gain: 0,
      count: 0
    }))
  }));

  // ========================================
  // WALK FORWARD VALIDATION
  // ========================================

  // Bat dau kiem nghiem tu phien thu 8
  // de hoat dong khi chi co 15 phien.

  for (let i = 8; i < s.length; i++) {
    const actual = s[i] === 't';

    for (const state of states) {
      for (const item of state.windows) {

        const prefix = s.slice(
          Math.max(0, i - item.size),
          i
        );

        const p = clip(
          state.model(prefix)
        );

        const likelihood =
          actual ? p : 1 - p;

        item.gain =
          forget * item.gain +
          Math.log(2 * likelihood);

        item.count =
          forget * item.count + 1;
      }
    }
  }

  // ========================================
  // ADAPTIVE WINDOW EVALUATION
  // ========================================

  const evaluated = states.map(state => {

    const candidates = state.windows.map(
      item => {

        const data = s.slice(
          -item.size
        );

        const p = clip(
          state.model(data)
        );

        const quality =
          item.gain /
          (item.count + 8);

        const weight = Math.exp(
          Math.max(
            -5,
            Math.min(
              5,
              quality * 12
            )
          )
        );

        return {
          size: item.size,
          p,
          quality,
          weight
        };
      }
    );

    const total = candidates.reduce(
      (sum, item) =>
        sum + item.weight,
      0
    );

    const probability =
      candidates.reduce(
        (sum, item) =>
          sum +
          item.p *
          item.weight / total,
        0
      );

    const quality =
      candidates.reduce(
        (sum, item) =>
          sum +
          item.quality *
          item.weight / total,
        0
      );

    return {
      model: state.model.name,

      probability,

      quality,

      windows: candidates.map(
        item => ({
          size: item.size,

          probabilityTai: item.p,

          quality: item.quality,

          weight:
            item.weight / total
        })
      )
    };
  });

  // ========================================
  // ADAPTIVE MODEL WEIGHTS
  // ========================================

  const rawWeights = evaluated.map(
    item => Math.exp(
      Math.max(
        -5,
        Math.min(
          5,
          item.quality * 12
        )
      )
    )
  );

  const totalWeight = rawWeights.reduce(
    (sum, w) => sum + w,
    0
  );

  let taiPoints = 0;
  let xiuPoints = 0;

  // ========================================
  // SYMMETRIC MODEL POINT SCORING
  // ========================================

  const modelPoints = evaluated.map(
    (item, index) => {

      const weight =
        rawWeights[index] /
        totalWeight;

      const signal =
        2 * item.probability - 1;

      const signed =
        100 * weight * signal;

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
        model: item.model,

        probabilityTai:
          item.probability,

        weight,

        quality: item.quality,

        tai,

        xiu,

        windows: item.windows
      };
    }
  );

  // ========================================
  // FINAL DECISION
  // ========================================

  const net =
    taiPoints - xiuPoints;

  const side =
    Math.abs(net) > 1e-10

      ? (net > 0 ? 't' : 'x')

      : (s.at(-1) === 't' ? 'x' : 't');

  // ========================================
  // MODEL QUALITY
  // ========================================

  const quality = modelPoints.reduce(
    (sum, item) =>
      sum +
      item.quality *
      item.weight,
    0
  );

  const strength = Math.min(
    1,
    Math.abs(net) / 100
  );

  // ========================================
  // NONLINEAR SIGNAL SCORE 51-99%
  // ========================================

  // Giu nguyen cong thuc V6.
  // Khong random.
  // Diem tin hieu khong phai xac suat thang.

  const smooth =
    Math.log1p(4 * strength) /
    Math.log(5);

  const confidence = Math.max(
    51,
    Math.min(
      99,
      Math.round(
        51 + 48 * smooth
      )
    )
  );

  // ========================================
  // STATISTICAL RELIABILITY
  // ========================================

  const positiveQuality = Math.max(
    0,
    quality
  );

  const reliability =
    positiveQuality /
    (positiveQuality + 0.05);

  const probabilityTai = clip(
    0.5 +
    net * reliability / 200
  );

  return {
    side,

    confidence,

    scoreType: 'model_signal_score',

    probabilityTai,

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

  const latestSource =
    sessions.at(-1).id;

  if (latestSource < last) {
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

  // Khong co phien moi
  // thi khong tinh lai du doan.

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
  // MODEL MIGRATION
  // ========================================

  if (migration) {
    next.modelVersion = VERSION;

    // Giu du lieu da tich luy
    // tu cac phien ban V2 den V6.

    next.modelStart =
      [
        'six-points-v2',
        'six-points-v3',
        'six-points-v4',
        'six-points-v5',
        'six-points-v6'
      ].includes(db.modelVersion) &&
      Number.isSafeInteger(db.modelStart)

        ? db.modelStart

        : sessions[0].id;

    next.history = next.history.filter(
      r => r.id < sessions[0].id
    );

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
  // CHECK PREVIOUS PREDICTION
  // ========================================

  for (const r of incoming) {
    const old =
      next.predictions[String(r.id)];

    if (old) {
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
  }

  // ========================================
  // FIFO STORAGE
  // ========================================

  // 15 phien: bat dau phan tich.
  // 16-200: tiep tuc tich luy.
  // 201: xoa phien cu nhat.
  // Luon giu toi da 200 phien.

  next.history =
    next.history.slice(-MAX);

  const latest =
    next.history.at(-1).id;

  const key =
    String(latest + 1);

  const pending =
    next.predictions[key];

  // ========================================
  // GENERATE NEXT PREDICTION
  // ========================================

  const analysis = recent(
    next.history,
    next.modelStart
  );

  // Chi du doan khi co it nhat 15 phien.
  // Giu nguyen du doan da cong bo.

  const prediction =
    analysis.length >= MIN_WINDOW

      ? (pending || predict(analysis))

      : null;

  next.predictions = prediction
    ? { [key]: prediction }
    : {};

  commit(next);

  // ========================================
  // CONSOLE LOG
  // ========================================

  if (prediction) {
    console.log(
      '================================'
    );

    console.log(
      '[PHIEN]',
      latest
    );

    console.log(
      '[KET QUA]',
      name(
        next.history.at(-1).result
      )
    );

    console.log(
      '[DU DOAN]',
      name(prediction.side)
    );

    console.log(
      '[DIEM TIN HIEU]',
      scorePct(prediction.confidence)
    );

    console.log(
      '[DIEM TAI]',
      prediction.taiPoints?.toFixed(4)
    );

    console.log(
      '[DIEM XIU]',
      prediction.xiuPoints?.toFixed(4)
    );

    console.log(
      '[PHAN TICH]',
      analysis.length + '/200'
    );

    console.log(
      '[LICH SU]',
      next.history.length + '/200'
    );

    console.log(
      '[WINDOWS]',
      prediction.analysisWindows
    );

    console.log(
      '================================'
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
// AUTOMATIC API FETCH
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
      '================================'
    );

    console.log('LC79 MD5 API ONLINE');
    console.log('VERSION:', VERSION);
    console.log('PORT:', PORT);
    console.log('API ID:', API_ID);
    console.log('MODELS:', models.length);
    console.log('MIN ANALYSIS:', MIN_WINDOW);
    console.log('MAX ANALYSIS:', WINDOW);
    console.log('WINDOWS:', WINDOWS);
    console.log('MAX HISTORY:', MAX);
    console.log('STORAGE: FIFO');
    console.log('CHECK INTERVAL: 5s');
    console.log('RANDOM: DISABLED');

    console.log(
      '================================'
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
