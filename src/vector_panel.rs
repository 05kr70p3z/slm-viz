use egui::{Color32, RichText, ScrollArea, Ui, Vec2};
use egui_plot::{Legend, Plot, PlotPoint};

use crate::model::{cosine_similarity, pca_2d, EmbeddingResult};

pub struct VectorPanel {
    pub show_raw_vectors: bool,
    pub selected_token_idx: usize,
    pub compare_token_idx: usize,
    pub view_mode: ViewMode,
}

#[derive(PartialEq)]
pub enum ViewMode {
    Pca2D,
    Heatmap,
}

impl Default for VectorPanel {
    fn default() -> Self {
        Self {
            show_raw_vectors: false,
            selected_token_idx: 0,
            compare_token_idx: 1,
            view_mode: ViewMode::Pca2D,
        }
    }
}

impl VectorPanel {
    pub fn show(
        &mut self,
        ui: &mut Ui,
        embedding: Option<&EmbeddingResult>,
        is_loading: bool,
    ) {
        ui.heading("Embedding Vector Visualization");
        ui.label("See how tokens become high-dimensional vectors and how they relate to each other.");

        ui.add_space(8.0);

        if is_loading {
            ui.spinner();
            ui.label("Computing embeddings...");
            return;
        }

        let emb = match embedding {
            Some(e) => e,
            None => {
                ui.label("Enter text in the Tokenizer tab first to generate embeddings.");
                return;
            }
        };

        let display_tokens: Vec<(usize, &str)> = emb
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.text.starts_with('['))
            .map(|(i, t)| (i, t.text.as_str()))
            .collect();

        if display_tokens.is_empty() {
            ui.label("No tokens to visualize (only special tokens found).");
            return;
        }

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view_mode, ViewMode::Pca2D, "2D PCA Projection");
            ui.selectable_value(&mut self.view_mode, ViewMode::Heatmap, "Similarity Heatmap");
            ui.checkbox(&mut self.show_raw_vectors, "Show raw vector values");
        });

        ui.add_space(4.0);
        ui.label(format!(
            "Embedding dimension: {}  |  {} tokens displayed",
            emb.token_embeddings.first().map_or(0, |v| v.len()),
            display_tokens.len(),
        ));

        ui.add_space(8.0);

        match self.view_mode {
            ViewMode::Pca2D => self.show_pca_view(ui, emb, &display_tokens),
            ViewMode::Heatmap => self.show_heatmap_view(ui, emb, &display_tokens),
        }
    }

    fn show_pca_view(&mut self, ui: &mut Ui, emb: &EmbeddingResult, tokens: &[(usize, &str)]) {
        let vecs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|(i, _)| emb.token_embeddings[*i].clone())
            .collect();

        let proj = pca_2d(&vecs, 2);

        let points: Vec<PlotPoint> = proj.iter().map(|v| PlotPoint::new(v[0] as f64, v[1] as f64)).collect();
        let labels: Vec<&str> = tokens.iter().map(|(_, t)| *t).collect();

        let height = (ui.available_height() * 0.6).max(200.0);

        Plot::new("pca_plot")
            .legend(Legend::default())
            .height(height)
            .allow_drag(true)
            .allow_zoom(true)
            .allow_scroll(true)
            .show(ui, |plot_ui| {
                for (i, point) in points.iter().enumerate() {
                    let hue = (i as f32 * 0.618033988749895) % 1.0;
                    let rgb = hsl_to_rgb(hue, 0.7, 0.6);
                    let color = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);

                    let points = egui_plot::Points::new([point.x, point.y])
                        .radius(8.0)
                        .color(color)
                        .name(labels[i].to_string());

                    plot_ui.points(points);

                    let _offset_x = labels[i].len() as f64 * 2.5;
                    let text = egui_plot::Text::new(
                        *point,
                        RichText::new(labels[i]).size(10.0).color(Color32::WHITE),
                    );
                    plot_ui.text(text);
                }
            });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Inspect token:");
            let token_names: Vec<&str> = tokens.iter().map(|(_, t)| *t).collect();
            if self.selected_token_idx >= token_names.len() {
                self.selected_token_idx = 0;
            }
            if self.compare_token_idx >= token_names.len() {
                self.compare_token_idx = 1.min(token_names.len().saturating_sub(1));
            }

            egui::ComboBox::from_id_salt("token_a")
                .selected_text(token_names.get(self.selected_token_idx).copied().unwrap_or("?"))
                .show_ui(ui, |ui| {
                    for (i, name) in token_names.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_token_idx, i, *name);
                    }
                });

            ui.label("vs");

            egui::ComboBox::from_id_salt("token_b")
                .selected_text(token_names.get(self.compare_token_idx).copied().unwrap_or("?"))
                .show_ui(ui, |ui| {
                    for (i, name) in token_names.iter().enumerate() {
                        ui.selectable_value(&mut self.compare_token_idx, i, *name);
                    }
                });
        });

        let idx_a = tokens[self.selected_token_idx].0;
        let idx_b = tokens[self.compare_token_idx].0;
        let sim = cosine_similarity(
            &emb.token_embeddings[idx_a],
            &emb.token_embeddings[idx_b],
        );

        ui.label(format!(
            "Cosine similarity between \"{}\" and \"{}\": {:.4}",
            tokens[self.selected_token_idx].1,
            tokens[self.compare_token_idx].1,
            sim,
        ));

        if sim > 0.7 {
            ui.label(RichText::new("  Very similar!").color(Color32::GREEN));
        } else if sim > 0.4 {
            ui.label(RichText::new("  Moderately similar").color(Color32::YELLOW));
        } else if sim > 0.0 {
            ui.label(RichText::new("  Weakly similar").color(Color32::LIGHT_RED));
        } else {
            ui.label(RichText::new("  Unrelated or opposite").color(Color32::RED));
        }

        if self.show_raw_vectors {
            ui.add_space(8.0);
            ui.separator();
            ui.collapsing("Raw vector (first 32 dims)", |ui| {
                let v = &emb.token_embeddings[idx_a];
                let display_dims = v.len().min(32);
                ui.label(
                    RichText::new(
                        v[..display_dims]
                            .iter()
                            .map(|x| format!("{:.4}", x))
                            .collect::<Vec<_>>()
                            .join(", "),
                    )
                    .monospace()
                    .size(10.0),
                );
            });
        }

        ui.add_space(16.0);

        ui.collapsing("Sentence Embedding (mean pooling)", |ui| {
            let dims = emb.sentence_embedding.len().min(32);
            ui.label(
                RichText::new(
                    emb.sentence_embedding[..dims]
                        .iter()
                        .map(|x| format!("{:.4}", x))
                        .collect::<Vec<_>>()
                        .join(", "),
                )
                .monospace()
                .size(10.0),
            );
        });

        ui.collapsing("Top-5 token pairs by similarity", |ui| {
            let n = tokens.len();
            let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let sim = cosine_similarity(
                        &emb.token_embeddings[tokens[i].0],
                        &emb.token_embeddings[tokens[j].0],
                    );
                    pairs.push((i, j, sim));
                }
            }
            pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            for (i, j, sim) in pairs.iter().take(5) {
                ui.label(format!(
                    "  \"{}\" ↔ \"{}\": {:.4}",
                    tokens[*i].1, tokens[*j].1, sim
                ));
            }
        });
    }

    fn show_heatmap_view(&mut self, ui: &mut Ui, emb: &EmbeddingResult, tokens: &[(usize, &str)]) {
        let n = tokens.len().min(30);
        let mut similarities = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in 0..n {
                similarities[i][j] = cosine_similarity(
                    &emb.token_embeddings[tokens[i].0],
                    &emb.token_embeddings[tokens[j].0],
                );
            }
        }

        ScrollArea::both().show(ui, |ui| {
            ui.label(format!("Cosine similarity matrix ({}×{})", n, n));
            ui.add_space(4.0);

            let cell_size = 22.0;
            let total_size = cell_size * (n + 1) as f32;
            let (_response, painter) =
                ui.allocate_painter(Vec2::new(total_size, total_size), egui::Sense::hover());

            for i in 0..n {
                let label = tokens[i].1;
                painter.text(
                    egui::pos2(1.0, (i as f32 + 1.0) * cell_size + cell_size * 0.3),
                    egui::Align2::RIGHT_CENTER,
                    label,
                    egui::FontId::monospace(10.0),
                    Color32::WHITE,
                );
                painter.text(
                    egui::pos2((i as f32 + 1.0) * cell_size + cell_size * 0.5, 1.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::monospace(10.0),
                    Color32::WHITE,
                );
            }

            for i in 0..n {
                for j in 0..n {
                    let sim = similarities[i][j];
                    let x = (j as f32 + 1.0) * cell_size;
                    let y = (i as f32 + 1.0) * cell_size;
                    let rect = egui::Rect::from_min_size(
                        egui::pos2(x, y),
                        Vec2::new(cell_size, cell_size),
                    );

                    let color = if sim > 0.8 {
                        Color32::from_rgb(0, 180, 0)
                    } else if sim > 0.5 {
                        Color32::from_rgb(120, 200, 0)
                    } else if sim > 0.2 {
                        Color32::from_rgb(200, 180, 0)
                    } else if sim > -0.2 {
                        Color32::from_rgb(180, 120, 0)
                    } else {
                        Color32::from_rgb(180, 60, 60)
                    };

                    painter.rect_filled(rect, 1.0, color);
                }
            }
        });
    }
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [u8; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}
