# app.py
import streamlit as st
import torch
import torch.nn as nn
import json, re

# ---------------------------
# 1. Model
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                          bidirectional=True, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.rnn(packed)
        return (h, c)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_tok, hidden):
        emb = self.dropout(self.embedding(input_tok.unsqueeze(1))) # [B,1,E]
        out, hidden = self.rnn(emb, hidden)                       # [B,1,H]
        pred = self.fc_out(out.squeeze(1))                        # [B,V]
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, hid_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.h_proj = nn.Linear(hid_dim * 2, hid_dim)
        self.c_proj = nn.Linear(hid_dim * 2, hid_dim)
    def init_hidden(self, h, c):
        num_layers = self.encoder.rnn.num_layers
        fwd = h[num_layers - 1]
        bwd = h[-1]
        cf = c[num_layers - 1]
        cb = c[-1]
        h0 = torch.tanh(self.h_proj(torch.cat((fwd, bwd), dim=1)))
        c0 = torch.tanh(self.c_proj(torch.cat((cf, cb), dim=1)))
        h0 = h0.unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)
        return (h0, c0)

# ---------------------------
# 2. Load vocabs
# ---------------------------
with open("data/bpe/vocab.json") as f:
    vocab_tokens = json.load(f)
bpe_token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}
bpe_id_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}

with open("data/src_vocab.json") as f:
    src_vocab = json.load(f)
if isinstance(src_vocab, list):
    src_vocab = {tok: i for i, tok in enumerate(src_vocab)}

PAD_IDX_SRC = src_vocab["<pad>"]
PAD_IDX_TGT = bpe_token_to_id["<pad>"]
SOS_ID = bpe_token_to_id["<sos>"]
EOS_ID = bpe_token_to_id["<eos>"]
UNK_ID = bpe_token_to_id["<unk>"]

# ---------------------------
# 3. Helpers
# ---------------------------
MAX_SRC_LEN = 80

def normalize_urdu(text: str) -> str:
    """Normalize Urdu text by unifying characters and removing diacritics."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[یےئ]", "ی", text)     # unify ya forms
    text = re.sub(r"[ھۀ]", "ہ", text)     # unify heh forms
    text = re.sub(r"[ًٌٍَُِّْٰ]", "", text)  # strip diacritics
    text = re.sub(r"\s+", " ", text.strip())  # normalize spaces and strip
    return text

def encode_src(line):
    ids = [src_vocab.get(ch, src_vocab["<unk>"]) for ch in line]
    return [src_vocab["<sos>"]] + ids[:MAX_SRC_LEN-2] + [src_vocab["<eos>"]]

def bpe_decode(tokens):
    words = []; cur = ""
    for tok in tokens:
        if tok.endswith("</w>"):
            cur += tok.replace("</w>", ""); words.append(cur); cur = ""
        else: cur += tok
    if cur: words.append(cur)
    return " ".join(words)

def beam_decode(model, src, src_lens, beam_width=1, max_len=80):
    model.eval()
    with torch.no_grad():
        (h, c) = model.encoder(src, src_lens)
        hidden = model.init_hidden(h, c)
        beams = [([SOS_ID], hidden, 0.0)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for tokens, hid, score in beams:
                if tokens[-1] == EOS_ID:
                    completed.append((tokens, score))
                    continue
                inp = torch.tensor([tokens[-1]], device=src.device)
                pred, new_hid = model.decoder(inp, hid)
                log_probs = torch.log_softmax(pred, dim=-1)
                topk = torch.topk(log_probs, beam_width, dim=-1)
                for i in range(beam_width):
                    tok = topk.indices[0, i].item()
                    sc = score + topk.values[0, i].item()
                    new_beams.append((tokens + [tok], new_hid, sc))
            if not new_beams: break
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
        if not completed: completed = beams
        best = max(completed, key=lambda x: x[1] if isinstance(x[1], float) else x[2])
        return best[0]

def translate(model, urdu_text, device, beam_width=1, max_len=80):
    urdu_text = normalize_urdu(urdu_text)
    src_ids = torch.tensor([encode_src(urdu_text)], dtype=torch.long).to(device)
    src_lens = torch.tensor([src_ids.size(1)], dtype=torch.long).to(device)
    pred_ids = beam_decode(model, src_ids, src_lens, beam_width=beam_width, max_len=max_len)
    if pred_ids and pred_ids[0] == SOS_ID: pred_ids = pred_ids[1:]
    if EOS_ID in pred_ids: pred_ids = pred_ids[:pred_ids.index(EOS_ID)]
    toks = [bpe_id_to_token[i] for i in pred_ids if i in bpe_id_to_token]
    return bpe_decode(toks)

# ---------------------------
# 4. Load model (match training hyperparams!)
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENC_EMB = 512; DEC_EMB = 512; HID = 512; ENC_LAY = 2; DEC_LAY = 3; DROPOUT = 0.1

enc = Encoder(len(src_vocab), ENC_EMB, HID, ENC_LAY, DROPOUT, PAD_IDX_SRC)
dec = Decoder(len(bpe_token_to_id), DEC_EMB, HID, DEC_LAY, DROPOUT, PAD_IDX_TGT)
model = Seq2Seq(enc, dec, HID).to(DEVICE)
model.load_state_dict(torch.load("best_seq2seq.pt", map_location=DEVICE))
model.eval()

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("📖 Urdu → Roman Urdu Translator")
user_input = st.text_area("✍️ Urdu Input:", height=150)

if st.button("Translate"):
    if user_input.strip():
        roman = translate(model, user_input, DEVICE, beam_width=1)
        st.success(f"**Roman Urdu:** {roman}")
    else:
        st.warning("⚠️ Please enter some Urdu text.")