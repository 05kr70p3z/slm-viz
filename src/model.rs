use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const VOCAB: &[(&str, u32)] = &[
    ("[CLS]", 101), ("[SEP]", 102), ("[PAD]", 0), ("[UNK]", 100),
    ("the", 1996), ("a", 1037), ("an", 2019), ("is", 2003),
    ("was", 2001), ("are", 2024), ("be", 2022), ("in", 1999),
    ("on", 2006), ("at", 2012), ("by", 2011), ("for", 2005),
    ("with", 2007), ("from", 2013), ("to", 2000), ("of", 1997),
    ("and", 1998), ("or", 2030), ("but", 2021), ("not", 2025),
    ("this", 2023), ("that", 2008), ("it", 2009), ("I", 1045),
    ("you", 2017), ("he", 2002), ("she", 2016), ("we", 2057),
    ("they", 2027), ("quick", 3674), ("brown", 2828),
    ("fox", 4412), ("jump", 5587), ("##s", 2015),
    ("over", 2058), ("lazy", 7687), ("dog", 3899),
    ("cat", 4937), ("bird", 2347), ("fish", 3668),
    ("happy", 3407), ("sad", 3992), ("big", 2502),
    ("small", 2461), ("fast", 3643), ("slow", 4039),
    ("good", 2204), ("bad", 2493), ("new", 2047),
    ("old", 2217), ("first", 2034), ("last", 2098),
    ("day", 2154), ("night", 2746), ("time", 2051),
    ("year", 2095), ("man", 2158), ("woman", 2608),
    ("child", 2161), ("world", 2088), ("life", 2166),
    ("hand", 2192), ("eye", 2232), ("head", 2105),
    ("way", 2126), ("thing", 2159), ("place", 2172),
    ("house", 2160), ("food", 2336), ("water", 2300),
    ("run", 2416), ("walk", 2558), ("talk", 2534),
    ("eat", 2547), ("sleep", 2823), ("read", 2435),
    ("write", 2346), ("learn", 2945), ("teach", 7920),
    ("love", 2293), ("like", 2074), ("hate", 5229),
    ("make", 2191), ("take", 2202), ("give", 2455),
    ("get", 2131), ("go", 2175), ("come", 2150),
    ("see", 2156), ("think", 2222), ("know", 2113),
    ("say", 2043), ("tell", 2273), ("ask", 2259),
    ("##ing", 2076), ("##ed", 2072), ("##ly", 2079),
    ("##er", 2067), ("##est", 2091), ("un", 4895),
    ("re", 2134), ("pre", 3415), ("dis", 4913),
    ("pro", 2518), ("con", 2115), ("ex", 2221),
    ("sub", 3513), ("inter", 2349), ("trans", 4384),
    ("micro", 6347), ("macro", 10717), ("auto", 2897),
    ("bio", 4142), ("geo", 7270), ("photo", 5136),
    ("tele", 5235), ("graph", 4055), ("phone", 2645),
    ("scope", 5527), ("meter", 7427), ("ology", 10037),
    ("data", 2951), ("code", 3241), ("model", 3030),
    ("token", 4205), ("vector", 6821), ("embed", 24891),
    ("attention", 2799), ("layer", 3649), ("network", 3146),
    ("neural", 5739), ("learning", 3565), ("machine", 2550),
    ("algorithm", 16886), ("train", 2676), ("test", 2632),
    ("loss", 3786), ("gradient", 13921), ("weight", 4424),
    ("bias", 9983), ("activation", 12880), ("function", 2082),
    ("input", 2995), ("output", 2976), ("hidden", 4408),
    ("deep", 4325), ("shallow", 9910), ("wide", 3507),
    ("narrow", 8515), ("high", 2157), ("low", 2257),
    ("long", 2189), ("short", 2352), ("thick", 8536),
    ("thin", 6051), ("heavy", 2776), ("light", 2147),
    ("dark", 2744), ("bright", 5215), ("soft", 3906),
    ("hard", 2532), ("smooth", 6385), ("rough", 6751),
    ("sharp", 5951), ("dull", 9831), ("warm", 4697),
    ("cold", 3064), ("hot", 2983), ("dry", 4679),
    ("wet", 4837), ("clean", 3625), ("dirty", 7720),
    ("rich", 3216), ("poor", 3082), ("strong", 3164),
    ("weak", 5168), ("young", 2840), ("mature", 6494),
    ("simple", 3185), ("complex", 5122), ("easy", 3073),
    ("hard", 2532), ("true", 2993), ("false", 5185),
    ("math", 10567), ("science", 2628), ("art", 2112),
    ("music", 2365), ("history", 2305), ("language", 2323),
    ("english", 2145), ("spanish", 3745), ("french", 2480),
    ("german", 3112), ("chinese", 2278), ("japanese", 2477),
];

fn lookup_id(word: &str) -> u32 {
    VOCAB
        .iter()
        .find(|(w, _)| *w == word)
        .map(|(_, id)| *id)
        .unwrap_or(100)
}

const EMBEDDING_DIM: usize = 384;

#[derive(Clone)]
pub struct ModelState {
    rng: rand::rngs::StdRng,
    seed_map: Vec<(String, [f32; EMBEDDING_DIM])>,
}

pub struct TokenInfo {
    pub text: String,
    pub id: u32,
    pub start: usize,
    pub end: usize,
}

pub struct EmbeddingResult {
    pub tokens: Vec<TokenInfo>,
    pub token_embeddings: Vec<Vec<f32>>,
    pub sentence_embedding: Vec<f32>,
    pub attention_mask: Vec<f32>,
}

impl ModelState {
    pub fn new() -> Self {
        let rng = StdRng::from_entropy();
        let seed_map: Vec<_> = VOCAB
            .iter()
            .map(|(word, _)| {
                let mut seed = StdRng::seed_from_u64(
                    seahash::hash(word.as_bytes()),
                );
                let mut emb = [0.0f32; EMBEDDING_DIM];
                for v in &mut emb {
                    *v = seed.gen_range(-1.0..1.0);
                }
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                for v in &mut emb {
                    *v /= norm;
                }
                (word.to_string(), emb)
            })
            .collect();
        Self { rng, seed_map }
    }

    pub fn tokenize(&mut self, text: &str) -> Vec<TokenInfo> {
        let mut tokens: Vec<TokenInfo> = Vec::new();

        tokens.push(TokenInfo {
            text: "[CLS]".to_string(),
            id: 101,
            start: 0,
            end: 0,
        });

        let lower = text.to_lowercase();
        let byte_positions: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        let lower_chars: Vec<char> = lower.chars().collect();

        let char_count = text.chars().count();
        let mut ci = 0;

        while ci < char_count {
            let ch = text[byte_positions[ci]..].chars().next().unwrap();

            if ch.is_whitespace() {
                ci += 1;
                continue;
            }

            if ch.is_ascii_punctuation() {
                let ch_str = ch.to_string();
                let byte_start = byte_positions[ci];
                let byte_end = byte_start + ch.len_utf8();
                tokens.push(TokenInfo {
                    text: ch_str,
                    id: lookup_id(&ch.to_string()),
                    start: byte_start,
                    end: byte_end,
                });
                ci += 1;
                continue;
            }

            let word_ci_start = ci;
            while ci < char_count {
                let c = text[byte_positions[ci]..].chars().next().unwrap();
                if c.is_whitespace() || c.is_ascii_punctuation() {
                    break;
                }
                ci += 1;
            }

            let byte_start = byte_positions[word_ci_start];
            let byte_end = if ci < char_count {
                byte_positions[ci]
            } else {
                text.len()
            };

            let word: String = lower_chars[word_ci_start..ci]
                .iter()
                .filter(|c| !c.is_ascii_punctuation())
                .collect();

            let entry = VOCAB
                .iter()
                .find(|(w, _)| *w == word.as_str())
                .copied()
                .unwrap_or_else(|| {
                    let mut best: Option<(&str, u32)> = None;
                    for e in VOCAB {
                        if e.0.len() <= word.len() && word.starts_with(e.0) {
                            if best.map_or(true, |b| e.0.len() > b.0.len()) {
                                best = Some(*e);
                            }
                        }
                    }
                    best.unwrap_or(("[UNK]", 100))
                });

            tokens.push(TokenInfo {
                text: entry.0.to_string(),
                id: entry.1,
                start: byte_start,
                end: byte_end,
            });

            if entry.0 != "[UNK]" {
                let remaining = &word[entry.0.len().min(word.len())..];
                if !remaining.is_empty() {
                    let sub_id = self.rng.gen_range(2000..9000);
                    tokens.push(TokenInfo {
                        text: format!("##{}", remaining),
                        id: sub_id,
                        start: byte_start + entry.0.len(),
                        end: byte_end,
                    });
                }
            }
        }

        tokens.push(TokenInfo {
            text: "[SEP]".to_string(),
            id: 102,
            start: text.len(),
            end: text.len(),
        });

        tokens
    }

    pub fn embed(&mut self, text: &str) -> EmbeddingResult {
        let tokens = self.tokenize(text);
        let token_embeddings: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| self.get_embedding(&t.text))
            .collect();

        let attention_mask: Vec<f32> = tokens
            .iter()
            .map(|t| {
                if t.text == "[PAD]" {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();

        let sentence_embedding = mean_pooling_vec(&token_embeddings, &attention_mask);

        EmbeddingResult {
            tokens,
            token_embeddings,
            sentence_embedding,
            attention_mask,
        }
    }

    fn get_embedding(&mut self, token: &str) -> Vec<f32> {
        if let Some((_, emb)) = self.seed_map.iter().find(|(w, _)| w == token) {
            let mut vec = emb.to_vec();
            let noise: f32 = self.rng.gen_range(-0.02..0.02);
            for v in &mut vec {
                *v += noise;
            }
            vec
        } else {
            let mut emb = vec![0.0f32; EMBEDDING_DIM];
            for v in &mut emb {
                *v = self.rng.gen_range(-1.0..1.0);
            }
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            for v in &mut emb {
                *v /= norm;
            }
            emb
        }
    }
}

fn mean_pooling_vec(embeddings: &[Vec<f32>], attention_mask: &[f32]) -> Vec<f32> {
    let dim = embeddings.first().map_or(0, |e| e.len());
    let mut pooled = vec![0.0f32; dim];
    for (emb, &mask) in embeddings.iter().zip(attention_mask.iter()) {
        if mask > 0.5 {
            for (p, &v) in pooled.iter_mut().zip(emb.iter()) {
                *p += v;
            }
        }
    }
    let total = attention_mask.iter().filter(|&&m| m > 0.5).count() as f32;
    if total > 0.0 {
        for p in &mut pooled {
            *p /= total;
        }
    }
    pooled
}

pub fn pca_2d(vectors: &[Vec<f32>], n_components: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() {
        return vec![];
    }
    let n = vectors.len();
    let d = vectors[0].len();
    let mean: Vec<f32> = (0..d)
        .map(|j| vectors.iter().map(|v| v[j]).sum::<f32>() / n as f32)
        .collect();
    let centered: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| v.iter().zip(&mean).map(|(&x, &m)| x - m).collect())
        .collect();
    let mut cov = vec![vec![0.0f32; d]; d];
    for row in &centered {
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            cov[i][j] /= n as f32;
        }
    }
    let eigen = simple_power_iteration(&cov, n_components);
    let result: Vec<Vec<f32>> = centered
        .iter()
        .map(|row| {
            eigen
                .iter()
                .map(|ev| row.iter().zip(ev.iter()).map(|(&x, &w)| x * w).sum())
                .collect()
        })
        .collect();
    let mut mins = vec![f32::MAX; n_components];
    let mut maxs = vec![f32::MIN; n_components];
    for point in &result {
        for (i, &v) in point.iter().enumerate() {
            mins[i] = mins[i].min(v);
            maxs[i] = maxs[i].max(v);
        }
    }
    result
        .into_iter()
        .map(|point| {
            point
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if (maxs[i] - mins[i]).abs() < 1e-8 {
                        0.5
                    } else {
                        (v - mins[i]) / (maxs[i] - mins[i])
                    }
                })
                .collect()
        })
        .collect()
}

fn simple_power_iteration(cov: &[Vec<f32>], num: usize) -> Vec<Vec<f32>> {
    let d = cov.len();
    let mut components: Vec<Vec<f32>> = Vec::new();
    for _ in 0..num {
        let mut v: Vec<f32> = (0..d)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        for _ in 0..50 {
            let mut av = vec![0.0f32; d];
            for i in 0..d {
                av[i] = cov[i].iter().zip(&v).map(|(&c, &x)| c * x).sum();
            }
            for prev in &components {
                let dot: f32 = av.iter().zip(prev.iter()).map(|(&a, &p)| a * p).sum();
                for (a, &p) in av.iter_mut().zip(prev.iter()) {
                    *a -= dot * p;
                }
            }
            let norm: f32 = av.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                break;
            }
            for (vi, &avi) in v.iter_mut().zip(av.iter()) {
                *vi = avi / norm;
            }
        }
        components.push(v);
    }
    components
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        dot / (na * nb)
    }
}

mod seahash {
    pub fn hash(data: &[u8]) -> u64 {
        let mut state: u64 = 0x6eed0e9da4d94a4f;
        for &byte in data {
            state = state.wrapping_mul(0x9e3779b97f4a7c15);
            state ^= (byte as u64).wrapping_mul(0xbf58476d1ce4e5b9);
            state = state.wrapping_add(state.rotate_left(27));
        }
        state ^= state >> 33;
        state = state.wrapping_mul(0xff51afd7ed558ccd);
        state ^= state >> 33;
        state = state.wrapping_mul(0xc4ceb9fe1a85ec53);
        state ^= state >> 33;
        state
    }
}
