
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
const WINDOW = 60;
const INTERVAL = 5000;
const VERSION = 'six-points-v2';

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

const countT = s =>
  [...s].filter(c => c === 't').length;

const beta = (t, n, a = 3) =>
  (t + a) / (n + 2 * a);

// ==========================================
// DATABASE
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

function commit(next) {
  next.history = next.history.slice(-MAX);

  next.sequence = sequence(
    next.history
  );

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

// Chỉ phân tích dữ liệu mới, liên tục

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
    s.length
  );
}

// ==========================================
// ALGORITHM 02
// EWMA RECENT DATA
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

  return beta(t, n);
}

// ==========================================
// CONTEXT MATCHING CORE
// ==========================================

function contextCounts(s, k) {
  const suffix = s.slice(-k);

  let t = 0;
  let n = 0;

  for (let i = 0; i + k < s.length; i++) {
    if (s.slice(i, i + k) !== suffix) {
      continue;
    }

    n++;

    t += Number(
      s[i + k] === 't'
    );
  }

  return [t, n];
}

// ==========================================
// ALGORITHM 03
// MARKOV CHAIN
// ==========================================

function markov(s) {
  return beta(
    ...contextCounts(s, 1)
  );
}

// ==========================================
// ALGORITHM 04
// VARIABLE ORDER MARKOV
// ==========================================

function variableMarkov(s) {
  let p = markov(s);

  const maxOrder = Math.min(
    6,
    s.length - 1
  );

  for (let k = 2; k <= maxOrder; k++) {
    const [t, n] = contextCounts(s, k);

    p = (t + 8 * p) / (n + 8);
  }

  return p;
}

// ==========================================
// ALGORITHM 05
// RUN LENGTH ANALYSIS
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

  for (let i = 0; i < s.length - 1; i++) {
    run =
      i > 0 && s[i] === s[i - 1]
        ? run + 1
        : 1;

    if (
      s[i] === side &&
      Math.min(run, 5) === Math.min(target, 5)
    ) {
      n++;

      t += Number(
        s[i + 1] === 't'
      );
    }
  }

  return beta(t, n);
}

// ==========================================
// ALGORITHM 06
// PATTERN RECOGNITION ENGINE
// ==========================================

// Nhan dien:
// Cau bet
// Cau 1-1
// Cau 2-1
// Cau 1-2
// Cau 2-2
// Cau 3-3
// Cau 3-1
// Cac chu ky lap tu 1-6 ky tu

function patternRecognition(s) {
  const hits = new Map();

  for (
    let period = 1;
    period <= 6;
    period++
  ) {
    const k = 2 * period;

    if (s.length < k + 1) {
      continue;
    }

    const block = s.slice(-period);

    const pattern = block.repeat(2);

    // Kiem tra mau cau dang lap

    if (!s.endsWith(pattern)) {
      continue;
    }

    // Loai bo chu ky trung lap

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

    // Doi chieu mau voi du lieu gan day

    for (
      let i = 0;
      i + k < s.length;
      i++
    ) {
      if (
        s.slice(i, i + k) !== pattern
      ) {
        continue;
      }

      const j = i + k;

      const age = s.length - 1 - j;

      const w = 2 ** (-age / 16);

      // Khong dem trung ket qua

      hits.set(
        j,
        Math.max(
          hits.get(j) || 0,
          w
        )
      );
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

  return beta(t, n);
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
// ADAPTIVE WEIGHTED POINT SCORING
// ==========================================

function predict(input) {
  // Chi su dung 60 phien moi nhat

  const s = input.slice(-WINDOW);

  const losses = models.map(
    () => 0
  );

  let effective = 0;

  const forget = 2 ** (-1 / 20);

  // ========================================
  // WALK FORWARD VALIDATION
  // ========================================

  for (let i = 4; i < s.length; i++) {
    const prefix = s.slice(0, i);

    const actual = s[i] === 't';

    effective =
      forget * effective + 1;

    models.forEach((model, j) => {
      const p = clip(
        model(prefix)
      );

      losses[j] =
        forget * losses[j] -
        Math.log(
          actual ? p : 1 - p
        );
    });
  }

  // ========================================
  // ADAPTIVE MODEL WEIGHTS
  // ========================================

  const min = Math.min(
    ...losses
  );

  const weights = losses.map(loss =>
    Math.exp(
      -4 * (loss - min) /
      (effective + 4)
    )
  );

  const totalWeight = weights.reduce(
    (a, b) => a + b,
    0
  );

  // ========================================
  // POINT SCORING SYSTEM
  // ========================================

  let taiPoints = 0;
  let xiuPoints = 0;

  const modelPoints = models.map(
    (model, i) => {

      const p = clip(
        model(s)
      );

      const weight =
        weights[i] / totalWeight;

      const signed =
        100 * weight * (2 * p - 1);

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
        model: model.name,
        tai,
        xiu,
        weight
      };
    }
  );

  // ========================================
  // FINAL PREDICTION
  // ========================================

  const net =
    taiPoints - xiuPoints;

  const side =
    net >= 0 ? 't' : 'x';

  const strength = Math.min(
    1,
    Math.abs(net) / 100
  );

  // ========================================
  // SIGNAL SCORE: 51-99%
  // INTEGER ONLY
  // ========================================

  // Diem tin hieu mo hinh.
  // Khong phai xac suat thang thuc te.

  const confidence = Math.round(
    51 + 48 * strength
  );

  return {
    side,

    confidence,

    scoreType: 'model_signal_score',

    probabilityTai:
      0.5 + net / 200,

    taiPoints,

    xiuPoints,

    modelPoints
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

  // Khong co phien moi thi im lang

  if (
    !migration &&
    !incoming.length &&
    db.predictions[String(last + 1)]
  ) {
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

  // Khoi tao du lieu moi cho model moi

  if (migration) {
    next.modelVersion = VERSION;

    next.modelStart = sessions[0].id;

    next.history = next.history.filter(
      r => r.id < next.modelStart
    );

    const merged = new Map(
      next.history.map(r => [r.id, r])
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
    }
  }

  // ========================================
  // FIFO MAX 200 SESSIONS
  // ========================================

  next.history =
    next.history.slice(-MAX);

  const latest =
    next.history.at(-1).id;

  const key =
    String(latest + 1);

  const pending =
    next.predictions[key];

  // Du doan bang du lieu moi

  next.predictions = {
    [key]: pending || predict(
      recent(
        next.history,
        next.modelStart
      )
    )
  };

  commit(next);

  if (incoming.length) {
    console.log(
      '[NEW]',
      latest,
      name(next.predictions[key].side),
      scorePct(
        next.predictions[key].confidence
      )
    );
  }

  return true;
}

// ==========================================
// AUTO FETCH API
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
// MAIN ENDPOINT
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

      ? pct(100 * wins / rows.length)

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
      'LC79 MD5 API ONLINE:',
      PORT
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
