const express = require('express');
const fs = require('node:fs');
const path = require('node:path');

const app = express();

const DATA_FILE = path.join(__dirname, 'lichsulc79.json');
const LEGACY_FILE = path.join(__dirname, 'lichsulc79md5.json');

const SOURCE_API = process.env.SOURCE_API ||
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0';

const PORT = process.env.PORT || 3000;
const POLL_INTERVAL = 15000;

app.use(express.json());

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// Chuẩn hóa dữ liệu phiên

function integer(value) {
  const isNumericString =
    typeof value === 'string' && /^\d+$/.test(value.trim());

  if (typeof value !== 'number' && !isNumericString) return null;

  const n = Number(value);
  return Number.isSafeInteger(n) ? n : null;
}

function label(value) {
  const s = String(value ?? '')
    .trim()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toUpperCase();

  if (s === 'TAI' || s === 'BIG') return 1;
  if (s === 'XIU' || s === 'SMALL') return 0;
  return null;
}

function normalize(raw, fromDisk = false) {
  if (!raw || typeof raw !== 'object') return null;

  const id = integer(raw.id ?? raw.session ?? raw.issue);
  if (id === null || id < 1 || id >= Number.MAX_SAFE_INTEGER) {
    return null;
  }

  const faces = raw.dices ?? raw.dice ?? raw.result;
  let dices = [];

  if (Array.isArray(faces) && faces.length === 3) {
    const values = faces.map(integer);
    if (values.every(v => v >= 1 && v <= 6)) {
      dices = values;
    }
  }

  const total = integer(raw.sum);
  let sum = null;

  if (dices.length === 3) {
    sum = dices.reduce((a, b) => a + b, 0);
  } else if (total >= 3 && total <= 18) {
    sum = total;
  }

  let result =
    label(raw.resultTruyenThong) ??
    label(raw.bigSmall) ??
    label(raw.result);

  // Trong file lịch sử, 1 là Tài và 0 là Xỉu.
  if (fromDisk && (raw.result === 0 || raw.result === 1)) {
    result = raw.result;
  }

  // Giữ cách phân loại theo tổng của API cũ khi thiếu nhãn.
  if (result === null && sum !== null) {
    result = Number(sum >= 11);
  }

  // Phiên chưa có kết quả sẽ được xét lại trong lần đồng bộ sau.
  if (result === null) return null;

  return {
    id,
    _id: String(raw._id ?? id),
    dices,
    sum,
    result
  };
}

// Context Tree Weighting (CTW) cho chuỗi T/X, với mốc tham chiếu 50/50.
const MAX_DEPTH = 8;
const MAX_INPUT_LENGTH = 4096;

function contextNode() {
  return { tai: 0, count: 0, logStopOverSplit: 0, children: {} };
}

function mix(first, second, logOdds) {
  const weight = 1 / (1 + Math.exp(-logOdds));
  return weight * first + (1 - weight) * second;
}

function logLikelihood(pTai, symbol) {
  return symbol === 'T' ? Math.log(pTai) : Math.log1p(-pTai);
}

class TXSequence {
  constructor() {
    this.root = contextNode();
    this.suffix = '';
    this.lastId = null;
    this.logTreeOverFair = 0;
  }

  // Tính xác suất trước khi cập nhật. Truy vấn với symbol=null chỉ đọc cây.
  visit(node, depth, symbol) {
    const local = (node.tai + 0.5) / (node.count + 1);
    let prediction = local;

    if (depth < MAX_DEPTH) {
      const previous = this.suffix[this.suffix.length - 1 - depth];
      let split = 0.5;

      if (previous !== undefined) {
        let child = node.children[previous];

        if (!child && symbol !== null) {
          child = contextNode();
          node.children[previous] = child;
        }

        if (child) split = this.visit(child, depth + 1, symbol);
      }

      prediction = mix(local, split, node.logStopOverSplit);

      if (symbol !== null) {
        node.logStopOverSplit += logLikelihood(local, symbol) - logLikelihood(split, symbol);
      }
    }

    if (symbol !== null) {
      node.count++;
      if (symbol === 'T') node.tai++;
    }

    return prediction;
  }

  probabilityTai() {
    return mix(this.visit(this.root, 0, null), 0.5, this.logTreeOverFair);
  }

  append(symbol) {
    const prediction = this.visit(this.root, 0, symbol);
    this.logTreeOverFair += logLikelihood(prediction, symbol) + Math.LN2;
    this.suffix = (this.suffix + symbol).slice(-MAX_DEPTH);
  }

  observeSession(row) {
    if (row.id !== this.lastId + 1) this.suffix = '';
    this.append(row.result === 1 ? 'T' : 'X');
    this.lastId = row.id;
  }
}

// Lưu lịch sử và khôi phục khi khởi động

function save(rows) {
  const temp = DATA_FILE + '.' + process.pid + '.tmp';
  fs.writeFileSync(temp, JSON.stringify(rows), 'utf8');
  fs.renameSync(temp, DATA_FILE);
}

function load() {
  const migrate = !fs.existsSync(DATA_FILE);

  if (migrate && !fs.existsSync(LEGACY_FILE)) {
    save([]);
    return [];
  }

  const file = migrate ? LEGACY_FILE : DATA_FILE;
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));

  if (!Array.isArray(raw)) {
    throw new Error('Lịch sử phải là mảng JSON: ' + file);
  }

  const byId = new Map();
  let skipped = 0;

  for (const item of raw) {
    const row = normalize(item, true);

    // Mã cũ có thể ghi nhầm phiên thiếu dữ liệu thành Xỉu, tổng 0.
    if (!row || (migrate && row.sum === null)) {
      if (!migrate) {
        throw new Error('Bản ghi không hợp lệ trong ' + file);
      }

      skipped++;
      continue;
    }

    byId.set(row.id, row);
  }

  const rows = [...byId.values()].sort((a, b) => a.id - b.id);

  if (migrate) {
    save(rows);
    console.log(
      '[MIGRATE]', rows.length, 'phiên hợp lệ, bỏ qua',
      skipped, 'bản ghi thiếu dữ liệu'
    );
  }

  return rows;
}

let history = load();
const known = new Set(history.map(row => row.id));

let model = new TXSequence();
for (const row of history) {
  model.observeSession(row);
}

// Đồng bộ nguồn API

async function sync() {
  try {
    const res = await fetch(SOURCE_API, {
      headers: {
        Accept: 'application/json',
        'User-Agent': 'Mozilla/5.0',
        Referer: 'https://tele68.com/'
      },
      signal: AbortSignal.timeout(10000)
    });

    if (!res.ok) throw new Error('HTTP ' + res.status);

    const body = await res.json();
    const list = [
      body,
      body?.data,
      body?.sessions,
      body?.list,
      body?.data?.sessions,
      body?.data?.list
    ].find(Array.isArray);

    if (!list) throw new Error('API chưa trả về danh sách phiên');

    const batch = new Map();

    for (const raw of list) {
      const row = normalize(raw);

      if (row && !known.has(row.id) && !batch.has(row.id)) {
        batch.set(row.id, row);
      }
    }

    if (batch.size === 0) return;

    const added = [...batch.values()].sort((a, b) => a.id - b.id);
    const backfill =
      history.length > 0 && added[0].id < history.at(-1).id;

    const next = history.concat(added);
    if (backfill) next.sort((a, b) => a.id - b.id);

    // Nếu ghi lỗi, giữ trạng thái cũ để lần đồng bộ sau thử lại.
    save(next);

    // Khi có phiên đến muộn, phát lại lịch sử theo đúng thứ tự.
    if (backfill) model = new TXSequence();

    for (const row of backfill ? next : added) {
      model.observeSession(row);
    }

    for (const row of added) known.add(row.id);
    history = next;
  } catch (err) {
    console.error('[SYNC]', err.message);
  }
}

// API trả kết quả

function respond(req, res) {
  const uptime = new Date(Date.now() + 7 * 3600000)
    .toISOString()
    .replace('Z', '+07:00');

  let row = history.at(-1);
  let engine = model;

  // Chuỗi nhập tay có thể dùng ngay cả trước lần đồng bộ đầu tiên.
  const input = req.query.chuoi;

  if (input !== undefined) {
    if (typeof input !== 'string' || input.length > MAX_INPUT_LENGTH ||
        input.length === 0 || /[^TX]/i.test(input)) {
      return res.status(400).json({
        error: 'chuoi phải gồm 1–4096 ký tự T hoặc X.'
      });
    }

    const sequence = input.toUpperCase();
    engine = new TXSequence();

    for (const symbol of sequence) engine.append(symbol);

    row = {
      id: null,
      result: Number(sequence.endsWith('T')),
      dices: [],
      sum: null
    };
  }

  if (!row) {
    return res.json({
      id: 'noname09',
      error: 'Chưa có dữ liệu, chờ sync',
      uptime
    });
  }

  const pTai = engine.probabilityTai();

  const payload = {
    id: 'noname09',
    phienhientai: row.id,
    ketqua: row.result ? 'tai' : 'xiu',
    xucxac: row.dices,
    tong: row.sum,
    dudoanphien: row.id === null ? null : row.id + 1,
    dudoan: pTai >= 0.5 ? 'tai' : 'xiu',
    dotincay: Math.round(100 * Math.max(pTai, 1 - pTai)) + '%',
    uptime
  };

  // Giữ mảng ba mặt xúc xắc trên một dòng như mẫu.
  const json = JSON.stringify(payload, null, 2).replace(
    /"xucxac": \[\s*(\d),\s*(\d),\s*(\d)\s*\]/,
    '"xucxac": [$1, $2, $3]'
  );

  return res.type('application/json').send(json);
}

app.get(['/', '/predict'], respond);

app.get('/ping', (req, res) => {
  res.send('pong');
});

app.get('/history', (req, res) => {
  res.json({
    count: history.length,
    last20: history.slice(-20)
  });
});

// Chờ lần đồng bộ hiện tại kết thúc rồi mới hẹn lần tiếp theo.
async function poll() {
  await sync();
  setTimeout(poll, POLL_INTERVAL).unref();
}

app.listen(PORT, () => {
  console.log('[BOOT] Port', PORT, '| Đã nạp', history.length, 'phiên');
  void poll();
});
