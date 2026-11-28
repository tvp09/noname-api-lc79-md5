
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

const INTERVAL = 5000;
const VERSION = 'six-points-v9';

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

const sign = c =>
  c === 't' ? 1 : -1;

const decay = (age, half) =>
  2 ** (-age / half);

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
// BAYESIAN FREQUENCY ANALYSIS
// ==========================================

// Phan tich tan suat tong the.
// Su dung prior doi xung TAI/XIU.

function frequency(s) {
  return beta(
    countT(s),
    s.length,
    6
  );
}

// ==========================================
// ALGORITHM 02
// BAYESIAN CHANGE POINT MIXTURE
// ==========================================

// Phan tich su thay doi trang thai.
// Khong mac dinh cau hien tai se lap lai.
// So sanh nhieu do dai du lieu gan nhat.

function regime(s) {

  function logEvidence(t, n) {
    let v = 0;

    for (let j = 0; j < t; j++) {
      v += Math.log(j + 2);
    }

    for (let j = 0; j < n - t; j++) {
      v += Math.log(j + 2);
    }

    for (let j = 0; j < n; j++) {
      v -= Math.log(j + 4);
    }

    return v;
  }

  const n = s.length;
  const t = countT(s);

  const base = logEvidence(t, n);

  let sum = beta(t, n, 2);
  let total = 1;

  const cuts = [8, 15, 30, 60]
    .filter(k => k <= n - 8);

  for (const k of cuts) {
    const r = countT(
      s.slice(-k)
    );

    const gain =
      logEvidence(r, k) +
      logEvidence(t - r, n - k) -
      base;

    const w =
      Math.exp(
        Math.min(8, gain)
      ) *
      0.25 / cuts.length;

    sum += w * beta(r, k, 2);
    total += w;
  }

  return sum / total;
}

// ==========================================
// ALGORITHM 03
// ADAPTIVE CONTEXT TREE
// ==========================================

// Nhan dien ngu canh tu 1 den 8 phien.
// Khong mac dinh cau bet.
// Khong mac dinh cau dao 1-1.
// Hoc ket qua tiep theo cua tung ngu canh.

function contextTree(s) {
  let p = 0.5;

  for (
    let k = 1;
    k <= Math.min(8, s.length - 1);
    k++
  ) {
    let t = 0;
    let n = 0;

    const suffix = s.slice(-k);

    for (let j = k; j < s.length; j++) {
      if (
        s.slice(j - k, j) !== suffix
      ) {
        continue;
      }

      const w = decay(
        s.length - 1 - j,
        45
      );

      n += w;

      t += w * Number(
        s[j] === 't'
      );
    }

    const prior = 6 + k;

    p =
      (t + prior * p) /
      (n + prior);
  }

  return p;
}

// ==========================================
// ALGORITHM 04
// APPROXIMATE MOTIF RECOGNITION
// ==========================================

// Nhan dien mau cau tuong dong.
// Chap nhan mau dao nguoc TAI/XIU.
// Chap nhan sai khac toi da 1 vi tri.
// Mau khop chinh xac co trong so cao hon.

function motifs(s) {
  const k = 6;
  const n = s.length;

  if (n < k + 1) {
    return 0.5;
  }

  let t = 0;
  let total = 0;

  for (let j = k; j < n; j++) {
    let d = 0;

    for (let h = 1; h <= k; h++) {
      d += Number(
        s[j - h] !== s[n - h]
      );
    }

    const mirror = d > k / 2;

    d = Math.min(d, k - d);

    if (d > 1) {
      continue;
    }

    const w =
      decay(n - 1 - j, 45) *
      (d === 0 ? 1 : 0.2);

    total += w;

    t += w * Number(
      (s[j] === 't') !== mirror
    );
  }

  return beta(t, total, 5);
}

// ==========================================
// ALGORITHM 05
// REGULARIZED LAG REGRESSION
// ==========================================

// Hoc moi lien he giua 8 phien gan nhat.
// Trong so duoc cap nhat tuan tu.
// Khong su dung random.
// Khong mac dinh ket qua tiep theo.

function lagRegression(s) {
  const k = 8;

  const w = Array(k + 1).fill(0);

  const sigmoid = z =>
    1 / (1 + Math.exp(-z));

  for (let i = k; i < s.length; i++) {

    const x = [
      1,

      ...Array.from(
        { length: k },
        (_, j) => sign(s[i - j - 1])
      )
    ];

    const p = sigmoid(
      w.reduce(
        (a, v, j) =>
          a + v * x[j],
        0
      )
    );

    const error =
      Number(s[i] === 't') - p;

    for (let j = 0; j < w.length; j++) {
      w[j] +=
        0.08 *
        (
          error * x[j] -
          0.10 * w[j]
        );
    }
  }

  if (s.length < k) {
    return 0.5;
  }

  const x = [
    1,

    ...Array.from(
      { length: k },

      (_, j) =>
        sign(s[s.length - j - 1])
    )
  ];

  return sigmoid(
    w.reduce(
      (a, v, j) =>
        a + v * x[j],
      0
    )
  );
}

// ==========================================
// ALGORITHM 06
// ADAPTIVE PERIODIC EXPERTS
// ==========================================

// Phan tich chu ky tu 2 den 12.
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
// Mau cau hon hop
// Chu ky thuan va nghich.
//
// Khong mac dinh chu ky nao luon dung.

function periodic(s) {
  let total = 1;
  let signal = 0;

  for (let k = 2; k <= 12; k++) {

    if (s.length < 2 * k + 3) {
      continue;
    }

    let same = 0;
    let n = 0;
    let local = 0;
    let nl = 0;

    for (let j = k; j < s.length; j++) {

      const w = decay(
        s.length - 1 - j,
        40
      );

      const hit = Number(
        s[j] === s[j - k]
      );

      same += w * hit;
      n += w;

      if (j >= s.length - 2 * k) {
        local += hit;
        nl++;
      }
    }

    const edge =
      2 * beta(same, n, 5) - 1;

    const recentEdge =
      2 * beta(local, nl, 2) - 1;

    // Chu ky phai co tin hieu
    // lich su va tin hieu gan day
    // cung huong.

    if (edge * recentEdge <= 0) {
      continue;
    }

    const w =
      Math.abs(edge * recentEdge) *
      n / (n + 10);

    signal +=
      w *
      edge *
      sign(s[s.length - k]);

    total += w;
  }

  return 0.5 + signal / (2 * total);
}

// ==========================================
// EXACTLY SIX PREDICTION MODELS
// ==========================================

const models = [
  frequency,
  regime,
  contextTree,
  motifs,
  lagRegression,
  periodic
];

// ==========================================
// ADAPTIVE ENSEMBLE ENGINE V9
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

  const m = models.length;

  const fast = Array(m).fill(0);
  const slow = Array(m).fill(0);

  // Ma tran tuong dong tin hieu.
  // Giam viec nhieu mo hinh
  // cung lap lai mot nhan dinh.

  const cross = Array.from(
    { length: m },
    () => Array(m).fill(0)
  );

  let nf = 0;
  let ns = 0;

  const df = decay(1, 15);
  const ds = decay(1, 50);

  // ========================================
  // WALK FORWARD VALIDATION
  // ========================================

  // Kiem nghiem tuan tu.
  // Khong su dung du lieu tuong lai.
  // So sanh voi moc 50/50.

  for (let i = 8; i < s.length; i++) {

    const p = models.map(f =>
      clip(
        f(s.slice(0, i))
      )
    );

    const y = s[i] === 't';

    nf = df * nf + 1;
    ns = ds * ns + 1;

    for (let j = 0; j < m; j++) {

      const gain = Math.log(
        2 * (y ? p[j] : 1 - p[j])
      );

      fast[j] =
        df * fast[j] + gain;

      slow[j] =
        ds * slow[j] + gain;

      for (let k = 0; k < m; k++) {

        cross[j][k] =
          ds * cross[j][k] +

          (2 * p[j] - 1) *
          (2 * p[k] - 1);
      }
    }
  }

  // ========================================
  // MODEL QUALITY EVALUATION
  // ========================================

  const quality = fast.map(
    (g, j) =>

      0.65 * g / (nf + 8) +

      0.35 * slow[j] / (ns + 12)
  );

  // ========================================
  // REDUNDANCY PENALTY
  // ========================================

  // Hai mo hinh co tin hieu
  // qua giong nhau se bi giam
  // muc do anh huong tong hop.

  const raw = quality.map((q, j) => {

    let redundancy = 0;

    for (let k = 0; k < m; k++) {

      if (k === j) {
        continue;
      }

      const corr =
        cross[j][k] /

        Math.sqrt(
          Math.max(
            1e-12,
            cross[j][j] *
            cross[k][k]
          )
        );

      redundancy +=
        Math.max(0, corr) ** 2;
    }

    return Math.exp(
      Math.max(
        -8,
        Math.min(8, 12 * q)
      )
    ) / (1 + 0.5 * redundancy);
  });

  const total = raw.reduce(
    (a, b) => a + b,
    0
  );

  let taiPoints = 0;
  let xiuPoints = 0;

  // ========================================
  // ADAPTIVE MODEL POINT SCORING
  // ========================================

  const modelPoints = models.map(
    (f, j) => {

      const p = clip(f(s));

      const weight =
        raw[j] / total;

      const signed =
        100 *
        weight *
        (2 * p - 1);

      const tai =
        Math.max(0, signed);

      const xiu =
        Math.max(0, -signed);

      taiPoints += tai;
      xiuPoints += xiu;

      return {
        model: f.name,

        probabilityTai: p,

        weight,

        quality: quality[j],

        tai,

        xiu
      };
    }
  );

  // ========================================
  // FINAL DECISION
  // ========================================

  const net =
    taiPoints - xiuPoints;

  // Khong uu tien mac dinh TAI/XIU.

  const side =
    Math.abs(net) > 1e-10

      ? (net > 0 ? 't' : 'x')

      : (s.at(-1) === 't' ? 'x' : 't');

  // ========================================
  // STATISTICAL RELIABILITY
  // ========================================

  const skill = modelPoints.reduce(
    (a, v) =>
      a + v.weight * v.quality,
    0
  );

  const positive =
    Math.max(0, skill);

  const reliability =
    positive / (positive + 0.04);

  // ========================================
  // MODEL SIGNAL STRENGTH
  // ========================================

  const strength = Math.min(
    1,
    Math.abs(net) / 100
  );

  const agreement =
    Math.abs(net) /

    Math.max(
      1e-10,
      taiPoints + xiuPoints
    );

  // ========================================
  // SIGNAL SCORE 51-99%
  // ========================================

  // Diem tin hieu khong phai
  // xac suat thang thuc te.
  // Khong random.
  // Khong ep tao do tin cay cao.

  const confidence = Math.max(
    51,

    Math.min(
      99,

      Math.round(
        51 +
        48 *
        Math.min(
          1,
          strength / 0.65
        ) *
        Math.sqrt(agreement)
      )
    )
  );

  return {
    side,

    confidence,

    scoreType: 'model_signal_score',

    probabilityTai: clip(
      0.5 +
      net * reliability / 200
    ),

    taiPoints,

    xiuPoints,

    modelPoints,

    reliability,

    analyzedSessions: s.length,

    analysisWindows: [s.length]
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
    const result = r.resultTruyenThong;

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

  // Chi tinh lai khi co phien moi.

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
        'six-points-v7',
        'six-points-v8'
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

  // Bat dau du doan tu 15 phien.
  // Tu phien 201 xoa phien cu nhat.
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
