"""
Build phone_lookup.bin from RKN DEF-9xx.csv (Russian mobile codes registry).
Run: py scripts/build_phone_lookup.py
Output: app/src/main/assets/phone_lookup.bin

Binary format (big-endian):
  magic[4]="PLKU" + N[4] + Nop[2] + Nreg[2]
  string table: (Nop+Nreg) strings, each = 1-byte-len + UTF-8 bytes
  N x entry (interleaved): from_key[8] + to_key[8] + op_idx[2] + reg_idx[2]
  key = DEF * 10_000_000 + 7-digit-subscriber-number
"""
import io, os, re, struct

# ---------------------------------------------------------------------------
# Operator normalization
# ---------------------------------------------------------------------------
_OP_MAP = [
    (["МТС", "МОБИЛЬНЫЕ ТЕЛЕСИСТЕМЫ"],             "МТС"),
    (["ВЫМПЕЛКОМ", "BEELINE", "БИЛАЙН"],            "Билайн"),
    (["МЕГАФОН"],                                   "МегаФон"),
    (["Т2 МОБАЙЛ", "Т2МОБАЙЛ", "TELE2", "ТЕЛЕ2",
      "Т2-МОБАЙЛ", "ЕКАТЕРИНБУРГ-2000"],            "Tele2"),
    (["РОСТЕЛЕКОМ", "БАШИНФОРМСВЯЗЬ"],              "Ростелеком"),
    (["ТИНЬКОФФ", "ТИНКОФФ", "TINKOFF", "Т-МОБ"], "Т-Мобайл"),
    (["МОТИВ"],                                     "Мотив"),
    (["СКАЙЛИНК", "SKYLINK"],                       "СкайЛинк"),
    (["YOTA", "ЙОТА", "СКАРТЕЛ"],                   "Yota"),
    (["СБЕРБАНК-ТЕЛЕКОМ", "СБЕРБАНК ТЕЛЕКОМ"],      "СберМобайл"),
    (["ТРАНСТЕЛЕКОМ", "ТТК-"],                      "ТрансТелеКом"),
    (["ТАТТЕЛЕКОМ"],                                "ТатТелеКом"),
]

def normalize_operator(raw: str) -> str:
    up = raw.upper()
    for keywords, label in _OP_MAP:
        if any(kw in up for kw in keywords):
            return label
    cleaned = re.sub(r'^(ПАО|ООО|АО|ЗАО|ОАО)\s+', '', raw.strip())
    return cleaned.strip('"').strip("'").strip()

# ---------------------------------------------------------------------------
# Region normalization
# ---------------------------------------------------------------------------
_REGION_REPLACE = [
    ("Москва и Московская область",             "Москва и МО"),
    ("г. Санкт-Петербург и Ленинградская область", "СПб и ЛО"),
    ("Санкт-Петербург и Ленинградская область", "СПб и ЛО"),
    ("Ханты-Мансийский АО - Югра",              "Ханты-Мансийский АО"),
    ("Ямало-Ненецкий АО",                       "Ямало-Ненецкий АО"),
    ("Ненецкий АО",                             "Ненецкий АО"),
    ("Чувашская Республика",                    "Чувашия"),
]

def normalize_region(raw: str) -> str:
    r = raw.split("|")[0].strip()
    r = r.split("*")[0].strip()   # "Москва * Московская область" → "Москва"
    r = re.sub(r'^г\.\s+', '', r) # "г. Москва" → "Москва"
    r = r.replace(' - Кузбасс', '')
    for old, new in _REGION_REPLACE:
        r = r.replace(old, new)
    r = re.sub(r'\s+обл\.\s*$', ' обл.', r.strip())
    r = re.sub(r'\.{2,}', '.', r)  # убираем случайные двойные точки
    return r.strip()

# ---------------------------------------------------------------------------
# Read CSV
# ---------------------------------------------------------------------------
csv_path = "C:/tmp/def9.csv"
entries = []

with open(csv_path, 'rb') as f:
    text = f.read().decode('utf-8-sig')

for line in text.splitlines()[1:]:
    parts = line.split(';')
    if len(parts) < 6 or not parts[0].strip():
        continue
    try:
        def_code = int(parts[0])
        from_num = int(parts[1])
        to_num   = int(parts[2])
    except ValueError:
        continue
    op  = normalize_operator(parts[4].strip().strip('"'))
    reg = normalize_region(parts[5].strip())

    from_key = def_code * 10_000_000 + from_num
    to_key   = def_code * 10_000_000 + to_num
    entries.append((from_key, to_key, op, reg))

entries.sort(key=lambda e: e[0])

# String tables
op_list  = list(dict.fromkeys(e[2] for e in entries))
reg_list = list(dict.fromkeys(e[3] for e in entries))
op_idx   = {s: i for i, s in enumerate(op_list)}
reg_idx  = {s: i for i, s in enumerate(reg_list)}

assert len(op_list)  <= 0xFFFF
assert len(reg_list) <= 0xFFFF

# ---------------------------------------------------------------------------
# Binary encoding (interleaved: per-entry from+to+op+reg)
# ---------------------------------------------------------------------------
buf = io.BytesIO()
buf.write(b'PLKU')
buf.write(struct.pack('>I', len(entries)))
buf.write(struct.pack('>H', len(op_list)))
buf.write(struct.pack('>H', len(reg_list)))

for s in op_list + reg_list:
    enc = s.encode('utf-8')
    assert len(enc) <= 255, f"String too long: {s!r}"
    buf.write(struct.pack('B', len(enc)))
    buf.write(enc)

for (fk, tk, op, reg) in entries:
    buf.write(struct.pack('>QQ', fk, tk))
    buf.write(struct.pack('>HH', op_idx[op], reg_idx[reg]))

data = buf.getvalue()

# Log
log = open("C:/tmp/lookup_build.txt", "w", encoding="utf-8")
log.write(f"Entries: {len(entries)}\n")
log.write(f"Operators ({len(op_list)}): {op_list[:20]}\n")
log.write(f"Regions  ({len(reg_list)}): {reg_list[:20]}\n")
log.write(f"Binary size: {len(data)} bytes ({len(data)//1024} KB)\n")

# Spot check
import struct as s2, bisect
off = [0]
raw = data
def rb(n): v=raw[off[0]:off[0]+n]; off[0]+=n; return v
def ri(n):
    fmt={4:'>I',2:'>H',8:'>Q'}[n]
    return s2.unpack(fmt, rb(n))[0]
assert rb(4)==b'PLKU'
_N,_nop,_nrg = ri(4),ri(2),ri(2)
def rs(c):
    r=[]
    for _ in range(c):
        l=raw[off[0]]; off[0]+=1
        r.append(raw[off[0]:off[0]+l].decode('utf-8')); off[0]+=l
    return r
_ops=rs(_nop); _regs=rs(_nrg)
_fk=[]; _tk=[]; _oi=[]; _ri=[]
for _ in range(_N):
    _fk.append(ri(8)); _tk.append(ri(8)); _oi.append(ri(2)); _ri.append(ri(2))
def lk(e164):
    d=e164.lstrip('+')
    if not d.startswith('7') or len(d)<11: return None
    dc=int(d[1:4])
    if dc<900 or dc>999: return None
    sub=int(d[4:11])
    key=dc*10_000_000+sub
    lo,hi=0,_N-1
    while lo<=hi:
        mid=(lo+hi)//2
        if key<_fk[mid]: hi=mid-1
        elif key>_tk[mid]: lo=mid+1
        else: return _ops[_oi[mid]],_regs[_ri[mid]]
    return None
for num in ["+79622052314","+79161234567","+79001234567","+79999999999"]:
    log.write(f"  check {num} -> {lk(num)}\n")

out = "app/src/main/assets/phone_lookup.bin"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'wb') as f:
    f.write(data)
log.write(f"Written: {out}\n")
log.close()
