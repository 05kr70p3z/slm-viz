use egui::{CentralPanel, Context, SidePanel, TopBottomPanel};

use crate::model::{EmbeddingResult, ModelState, TokenInfo};
use crate::token_panel::TokenPanel;
use crate::vector_panel::VectorPanel;

#[derive(PartialEq)]
enum Tab {
    Tokenizer,
    Vectors,
}

pub struct SlmVizApp {
    model: ModelState,
    token_panel: TokenPanel,
    vector_panel: VectorPanel,
    active_tab: Tab,
    tokens: Option<Vec<TokenInfo>>,
    embedding: Option<EmbeddingResult>,
    last_text: String,
}

impl SlmVizApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut model = ModelState::new();
        let tokens = model.tokenize("The quick brown fox jumps over the lazy dog.");
        let embedding = model.embed("The quick brown fox jumps over the lazy dog.");

        Self {
            model,
            token_panel: TokenPanel::default(),
            vector_panel: VectorPanel::default(),
            active_tab: Tab::Tokenizer,
            tokens: Some(tokens),
            embedding: Some(embedding),
            last_text: "The quick brown fox jumps over the lazy dog.".to_string(),
        }
    }

    fn recompute(&mut self) {
        let text = self.token_panel.get_text().to_string();
        if text.is_empty() || text == self.last_text {
            return;
        }
        self.last_text = text.clone();
        self.tokens = Some(self.model.tokenize(&text));
        self.embedding = Some(self.model.embed(&text));
    }
}

impl eframe::App for SlmVizApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let text = self.token_panel.get_text().to_string();
        if text != self.last_text {
            self.recompute();
        }

        TopBottomPanel::top("title_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SLM Viz");
                ui.separator();
                ui.label("Tokenization & Embedding Visualizer");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("Simulated all-MiniLM-L6-v2 · 384-dim embeddings");
                });
            });
        });

        SidePanel::left("tabs")
            .min_width(120.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    ui.selectable_value(&mut self.active_tab, Tab::Tokenizer, "Tokenizer");
                    ui.selectable_value(&mut self.active_tab, Tab::Vectors, "Vectors");
                });
                ui.add_space(16.0);
                ui.separator();
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Visualize how SLMs turn\ntext into tokens and vectors.").small().weak());
                ui.add_space(4.0);
                ui.label(egui::RichText::new("~200 token vocab\n384-dim embeddings").small().weak());
            });

        CentralPanel::default().show(ctx, |ui| {
            match self.active_tab {
                Tab::Tokenizer => {
                    self.token_panel.show(
                        ui,
                        self.tokens.as_deref(),
                        false,
                        |_| {},
                    );
                }
                Tab::Vectors => {
                    self.vector_panel.show(ui, self.embedding.as_ref(), false);
                }
            }
        });
    }
}
