mod app;
mod model;
mod token_panel;
mod vector_panel;

use app::SlmVizApp;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("SLM Viz - Tokenization & Embedding Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "slm-viz",
        native_options,
        Box::new(|cc| Ok(Box::new(SlmVizApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast;

    console_error_panic_hook::set_once();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("no canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("element is not a canvas");

        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(SlmVizApp::new(cc)))),
            )
            .await
            .expect("failed to start eframe");
    });
}
