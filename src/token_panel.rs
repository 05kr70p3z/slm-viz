use egui::{Color32, RichText, ScrollArea, Ui};

use crate::model::TokenInfo;

pub struct TokenPanel {
    pub input_text: String,
    pub show_special_tokens: bool,
}

impl Default for TokenPanel {
    fn default() -> Self {
        Self {
            input_text: "The quick brown fox jumps over the lazy dog.".to_string(),
            show_special_tokens: true,
        }
    }
}

impl TokenPanel {
    pub fn show(
        &mut self,
        ui: &mut Ui,
        tokens: Option<&[TokenInfo]>,
        is_loading: bool,
        on_text_change: impl FnOnce(&str),
    ) {
        ui.heading("Tokenization Visualizer");
        ui.label("See how text gets split into tokens and mapped to token IDs.");

        ui.add_space(8.0);

        let response = ui.add_sized(
            [ui.available_width(), 80.0],
            egui::TextEdit::multiline(&mut self.input_text)
                .hint_text("Enter text to tokenize...")
                .desired_rows(3),
        );

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_special_tokens, "Show special tokens");
            if ui.button("Clear").clicked() {
                self.input_text.clear();
            }
        });

        if response.changed() {
            on_text_change(&self.input_text);
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        if is_loading {
            ui.spinner();
            ui.label("Loading model and tokenizing...");
            return;
        }

        let tokens = match tokens {
            Some(t) if !t.is_empty() => t,
            _ => {
                ui.label("Enter text above and press enter to tokenize.");
                return;
            }
        };

        ui.label(format!("{} tokens", tokens.len()));
        ui.add_space(4.0);

        ScrollArea::vertical()
            .max_height(ui.available_height() - 20.0)
            .show(ui, |ui| {
                let mut viz_tokens: Vec<&TokenInfo> = tokens.iter().collect();
                if !self.show_special_tokens {
                    viz_tokens.retain(|t| !t.text.starts_with('['));
                }

                for token in &viz_tokens {
                    let hue = (token.id as f32 * 0.618033988749895) % 1.0;
                    let rgb = hsl_to_rgb(hue, 0.7, 0.85);
                    let color = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);

                    ui.horizontal(|ui| {
                        let token_text = if token.text.starts_with("##") {
                            format!("{} (subword)", token.text.trim_start_matches("##"))
                        } else if token.text.starts_with('[') {
                            format!("{} (special)", token.text)
                        } else {
                            token.text.clone()
                        };

                        let (label_rect, _resp) = ui
                            .allocate_at_least(
                                egui::vec2(120.0, 28.0),
                                egui::Sense::hover(),
                            );

                        ui.painter().rect_filled(label_rect, 4.0, color);
                        ui.painter().text(
                            label_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            &token_text,
                            egui::FontId::monospace(14.0),
                            Color32::BLACK,
                        );

                        ui.add_sized(
                            [70.0, 28.0],
                            egui::Label::new(
                                RichText::new(format!("#{}", token.id))
                                    .monospace()
                                    .strong(),
                            ),
                        );

                        if token.start <= token.end && token.start < self.input_text.len() {
                            let snippet = &self.input_text[token.start..token.end.min(self.input_text.len())];
                            if !snippet.is_empty() {
                                ui.label(
                                    RichText::new(format!("\"{}\"", snippet))
                                        .italics()
                                        .weak(),
                                );
                            }
                        }
                    });

                    ui.add_space(2.0);
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.label("Token ID sequence:");
                let ids: Vec<String> = viz_tokens.iter().map(|t| t.id.to_string()).collect();
                ui.label(RichText::new(ids.join(" → ")).monospace());

                let total_ids: u32 = viz_tokens.iter().map(|t| t.id + 1).max().unwrap_or(0);
                let special_count = viz_tokens.iter().filter(|t| t.text.starts_with('[')).count();
                let subword_count = viz_tokens.iter().filter(|t| t.text.starts_with("##")).count();
                let normal_count = viz_tokens.len() - special_count - subword_count;

                ui.add_space(4.0);
                ui.label(format!(
                    "Vocabulary range: 0–{}, Normal: {}, Subwords: {}, Special: {}",
                    total_ids, normal_count, subword_count, special_count
                ));
            });
    }

    pub fn get_text(&self) -> &str {
        &self.input_text
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
