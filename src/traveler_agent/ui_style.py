"""Caribe Nocturno / UTB Claro presentation. No inference or remote assets."""
import gradio as gr

_PRIMARY = gr.themes.colors.teal
THEME = gr.themes.Soft(primary_hue=_PRIMARY, secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate, font=["Manrope", "DM Sans", "ui-sans-serif", "system-ui", "sans-serif"]).set(
    body_background_fill="#f3f7fa", body_background_fill_dark="#061217",
    block_background_fill="#ffffff", block_background_fill_dark="#0c1f27",
    body_text_color="#0b2239", body_text_color_dark="#eefaf9",
    background_fill_primary="#ffffff", background_fill_primary_dark="#0c1f27",
    background_fill_secondary="#eef4f8", background_fill_secondary_dark="#112a35")

# Native controls retain Gradio semantics, keyboard handling and real callbacks.
# These are the only client-side operations: theme attributes, never inference.
THEME_JS = """(value) => {
 const dark = value !== 'UTB Claro';
 document.documentElement.dataset.tiTheme = dark ? 'caribe' : 'utb';
 document.documentElement.classList.toggle('dark', dark);
 document.querySelectorAll('.gradio-container').forEach(el => el.classList.toggle('dark', dark));
 return [];
}"""
INITIAL_THEME_JS = """() => {
 document.documentElement.lang = 'es';
 document.documentElement.dataset.tiTheme = 'caribe';
 document.documentElement.classList.add('dark');
 document.querySelectorAll('.gradio-container').forEach(el => el.classList.add('dark'));
 return [];
}"""

UI_CSS = """
:root, :root[data-ti-theme="caribe"] {
 --ti-bg:#061217; --ti-panel:#0c1f27; --ti-panel2:#112a35; --ti-panel3:#09181f;
 --ti-edge:#1b3f4e; --ti-edge-strong:#27586c; --ti-muted:#9abac6; --ti-text:#eefaf9;
 --ti-soft:#cde4e6; --ti-primary:#0f766e; --ti-cyan:#5eead4; --ti-gold:#fcd34d;
 --ti-green:#6ee7b7; --ti-red:#fda4af; --ti-gold-bg:#2c2514; --ti-gold-edge:#796025;
 --ti-shadow:0 20px 60px #0004; --ti-summary:#0f2731; --ti-focus:#5eead4;
 --ti-hero:linear-gradient(95deg,#f0fdfa,#5eead4 52%,#fbbf24);
 color-scheme:dark;
}
:root[data-ti-theme="utb"] {
 --ti-bg:#f3f7fa; --ti-panel:#fff; --ti-panel2:#eef4f8; --ti-panel3:#f6f9fc;
 --ti-edge:#d5e2ec; --ti-edge-strong:#a8bfce; --ti-muted:#526a7e; --ti-text:#0b2239;
 --ti-soft:#26435e; --ti-primary:#0b4f8a; --ti-cyan:#087c72; --ti-gold:#92400e;
 --ti-green:#047857; --ti-red:#be123c; --ti-gold-bg:#fef3c7; --ti-gold-edge:#dba628;
 --ti-shadow:0 14px 38px #0b223914; --ti-summary:#eaf4f7; --ti-focus:#0b4f8a;
 --ti-hero:linear-gradient(95deg,#0b4f8a,#087c72 60%,#92400e);
 color-scheme:light;
}
body, .gradio-container {
 background:radial-gradient(ellipse 70% 40% at 85% 0%,#14b8a60d,transparent),var(--ti-bg)!important;
 color:var(--ti-text)!important; font-family:Manrope,'DM Sans',ui-sans-serif,system-ui,sans-serif!important;
}
.gradio-container, .gradio-container.dark {
 --body-background-fill:var(--ti-bg); --body-background-fill-dark:var(--ti-bg);
 --body-text-color:var(--ti-text); --body-text-color-dark:var(--ti-text);
 --body-text-color-subdued:var(--ti-muted); --body-text-color-subdued-dark:var(--ti-muted);
 --block-background-fill:var(--ti-panel); --block-background-fill-dark:var(--ti-panel);
 --background-fill-primary:var(--ti-panel); --background-fill-primary-dark:var(--ti-panel);
 --background-fill-secondary:var(--ti-panel2); --background-fill-secondary-dark:var(--ti-panel2);
 --border-color-primary:var(--ti-edge); --border-color-primary-dark:var(--ti-edge);
 --border-color-accent:var(--ti-edge-strong); --border-color-accent-dark:var(--ti-edge-strong);
 --block-border-color:var(--ti-edge); --block-border-color-dark:var(--ti-edge);
 --block-label-text-color:var(--ti-soft); --block-label-text-color-dark:var(--ti-soft);
 --block-title-text-color:var(--ti-text); --block-title-text-color-dark:var(--ti-text);
 --block-label-background-fill:transparent; --block-label-background-fill-dark:transparent;
 --block-title-background-fill:transparent; --block-title-background-fill-dark:transparent;
 --input-background-fill:var(--ti-panel3); --input-background-fill-dark:var(--ti-panel3);
 --input-border-color:var(--ti-edge-strong); --input-border-color-dark:var(--ti-edge-strong);
 --input-placeholder-color:var(--ti-muted); --input-placeholder-color-dark:var(--ti-muted);
 --button-secondary-background-fill:var(--ti-panel2); --button-secondary-background-fill-dark:var(--ti-panel2);
 --button-secondary-background-fill-hover:var(--ti-panel3); --button-secondary-background-fill-hover-dark:var(--ti-panel3);
 --button-secondary-text-color:var(--ti-soft); --button-secondary-text-color-dark:var(--ti-soft);
 --button-primary-background-fill:var(--ti-primary); --button-primary-background-fill-dark:var(--ti-primary);
 --button-primary-background-fill-hover:#0f766e; --button-primary-background-fill-hover-dark:#0f766e;
 --button-primary-text-color:#fff; --button-primary-text-color-dark:#fff;
 --checkbox-background-color:var(--ti-panel3); --checkbox-background-color-dark:var(--ti-panel3);
 --checkbox-background-color-selected:var(--ti-primary); --checkbox-background-color-selected-dark:var(--ti-primary);
 --checkbox-label-text-color:var(--ti-soft); --checkbox-label-text-color-dark:var(--ti-soft);
 --checkbox-label-background-fill:var(--ti-panel2); --checkbox-label-background-fill-dark:var(--ti-panel2);
 --checkbox-label-background-fill-selected:var(--ti-primary); --checkbox-label-background-fill-selected-dark:var(--ti-primary);
 --checkbox-label-text-color-selected:#fff; --checkbox-label-text-color-selected-dark:#fff;
 --table-even-background-fill:var(--ti-panel); --table-even-background-fill-dark:var(--ti-panel);
 --table-odd-background-fill:var(--ti-panel3); --table-odd-background-fill-dark:var(--ti-panel3);
 --table-text-color:var(--ti-soft); --table-text-color-dark:var(--ti-soft);
 --link-text-color:var(--ti-cyan); --link-text-color-dark:var(--ti-cyan);
 max-width:1500px!important; padding:0 32px 24px!important; min-height:100vh;
}
.gradio-container button:focus-visible, .gradio-container input:focus-visible,
.gradio-container textarea:focus-visible, .gradio-container summary:focus-visible,
.gradio-container [role="tab"]:focus-visible {
 outline:3px solid var(--ti-focus)!important; outline-offset:3px!important;
}
.gradio-container button {transition:background .18s,border-color .18s;min-height:42px;}
.gradio-container button.secondary {background:var(--ti-panel2)!important;color:var(--ti-soft)!important;
 border:1px solid var(--ti-edge-strong)!important;box-shadow:none!important;}
.gradio-container input[type="checkbox"],.gradio-container input[type="radio"] {border:1px solid var(--ti-muted)!important;}
.gradio-container textarea {line-height:1.7!important;}
.gradio-container label, .gradio-container .prose {color:var(--ti-soft);}
.gradio-container .prose p, .gradio-container .prose li {color:inherit;}
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3,
.gradio-container .prose h4 {color:var(--ti-text);}
#ti-topbar {align-items:center;border-bottom:1px solid var(--ti-edge);padding:20px 0 17px;gap:18px;}
.ti-brand {display:flex;gap:12px;align-items:center;}
.ti-brand-mark {display:grid;place-items:center;background:linear-gradient(140deg,#0d9488,#14b8a6);
 width:43px;height:43px;border-radius:13px;color:#fff;flex:none;box-shadow:0 8px 24px #14b8a629;}
.ti-brand svg {width:27px;height:27px;}
.ti-brand-name {font-size:19px;font-weight:800;letter-spacing:-.7px;color:var(--ti-text);}
.ti-brand-name span {color:var(--ti-cyan);}
.ti-brand-sub {font-size:10px;color:var(--ti-muted);letter-spacing:1.7px;font-weight:700;margin-top:4px;}
#ti-theme {background:transparent!important;border:none!important;padding:0;}
#ti-topbar .form {background:transparent!important;border:none!important;box-shadow:none!important;}
#ti-theme label {font-size:11px!important;}
#ti-nav {border:none;}
#ti-nav [role="tablist"] {display:flex;gap:6px;border-bottom:1px solid var(--ti-edge);padding:14px 0;}
#ti-nav [role="tab"] {border:1px solid transparent;border-radius:10px!important;font-weight:700;
 color:var(--ti-muted);padding:10px 20px!important;background:transparent;min-height:44px;}
#ti-nav [role="tab"][aria-selected="true"] {background:var(--ti-primary)!important;color:#fff!important;border-color:var(--ti-primary)!important;}
#ti-nav > .tabitem {border:none;padding:0;}
.ti-hero {display:flex;justify-content:space-between;gap:20px;align-items:end;margin:28px 0 25px;}
.ti-eyebrow {font-size:10px;letter-spacing:2.1px;font-weight:800;color:var(--ti-cyan);text-transform:uppercase;}
.ti-hero h1 {font-size:clamp(27px,2.8vw,38px);font-weight:800;line-height:1.2;letter-spacing:-1.4px;margin:12px 0 10px;}
.gradio-container .ti-hero h1 span {color:var(--ti-cyan)!important;}
.ti-hero p,.ti-copy {color:var(--ti-muted);font-size:13px;line-height:1.7;margin:0;max-width:680px;}
.ti-hero-tags {display:flex;flex-wrap:wrap;gap:8px;}
.ti-tag {border:1px solid var(--ti-edge-strong);border-radius:8px;padding:8px 10px;background:var(--ti-panel2);font-size:10px;font-weight:700;color:var(--ti-soft);white-space:nowrap;}
.ti-tag-accent {color:var(--ti-cyan);}
.ti-tag-gold {color:var(--ti-gold);background:var(--ti-gold-bg);border-color:var(--ti-gold-edge);}
#ti-analysis-row {align-items:start;gap:22px;}
.ti-pane {border:1px solid var(--ti-edge)!important;border-radius:18px!important;background:var(--ti-panel)!important;
 box-shadow:var(--ti-shadow);overflow:visible;padding:22px!important;min-width:0;}
.ti-pane-head {border-bottom:1px solid var(--ti-edge);padding-bottom:17px;margin-bottom:18px;}
.ti-pane-head h2 {font-size:16px;letter-spacing:-.4px;margin:0 0 5px;font-weight:800;color:var(--ti-text);}
.ti-pane-head p {font-size:12px;line-height:1.6;color:var(--ti-muted);margin:0;}
#ti-input-pane .block {box-shadow:none;}
#ti-input-pane #ti-qwen {background:var(--ti-panel2);border:1px solid var(--ti-edge);border-radius:10px;padding:12px;}
#ti-char-count {text-align:right;font-size:11px;color:var(--ti-muted);min-height:20px;}
.ti-example-note {color:var(--ti-muted);font-size:11px;margin:8px 0 2px;}
.ti-example-row {gap:6px!important;}
.ti-example-row button {font-size:11px!important;padding:6px 9px!important;min-width:0!important;}
.ti-input-note {padding:12px 14px;border:1px solid var(--ti-edge);border-radius:10px;background:var(--ti-panel3);font-size:11px;line-height:1.7;color:var(--ti-muted);}
.ti-input-note strong {color:var(--ti-soft);}
.ti-placeholder {min-height:330px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;padding:32px;color:var(--ti-muted);gap:12px;}
.ti-placeholder svg {width:46px;height:46px;color:var(--ti-cyan);}
.ti-placeholder h3 {font-size:17px;color:var(--ti-text);margin:0;}
.ti-placeholder p {font-size:12px;line-height:1.75;max-width:310px;margin:0;}
.ti-report {color:var(--ti-soft);font-size:13px;line-height:1.72;}
.ti-report h3,.ti-report h4,.ti-report p {margin:0;}
.ti-report-source {font-size:11px;color:var(--ti-muted);margin-bottom:14px!important;}
.ti-report-summary {background:radial-gradient(ellipse at 100% 0%,#14b8a615,transparent 65%),var(--ti-summary);
 border:1px solid var(--ti-edge-strong);padding:19px 21px;border-radius:12px;margin-bottom:16px;}
.ti-report-summary h4 {font-size:10px;letter-spacing:1.6px;text-transform:uppercase;color:var(--ti-cyan);margin-bottom:10px;}
.ti-report-summary p {font-size:15px;font-weight:600;line-height:1.8;color:var(--ti-text);}
.ti-indicators {display:flex;flex-wrap:wrap;gap:7px;margin-bottom:17px;}
.ti-indicator {font-size:10px;color:var(--ti-muted);border:1px solid var(--ti-edge);border-radius:8px;background:var(--ti-panel2);padding:8px 10px;}
.ti-indicator strong {color:var(--ti-text);margin-left:5px;}
.ti-sent-negativo strong {color:var(--ti-red);}.ti-sent-positivo strong {color:var(--ti-green);}
.ti-report-grid {display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;}
.ti-report-block {border:1px solid var(--ti-edge);border-top:3px solid var(--ti-cyan);background:var(--ti-panel3);
 border-radius:11px;padding:16px 17px;min-width:0;overflow-wrap:anywhere;}
.ti-report-block h4 {font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px;font-weight:800;color:var(--ti-cyan);}
.ti-report-problem {border-top-color:var(--ti-red);}.ti-report-problem h4 {color:var(--ti-red);}
.ti-report-rec {border-top-color:var(--ti-green);}.ti-report-rec h4 {color:var(--ti-green);}
.ti-report-action {border-top-color:var(--ti-gold);}.ti-report-action h4 {color:var(--ti-gold);}
.ti-report blockquote {margin:0 0 9px;border-left:2px solid var(--ti-cyan);padding-left:12px;font-size:12px;font-style:italic;line-height:1.7;}
.ti-warnings {margin-top:15px;border:1px solid var(--ti-gold-edge);background:var(--ti-gold-bg);border-radius:10px;padding:12px 15px;color:var(--ti-gold);font-size:11px;line-height:1.7;}
.ti-warnings h4 {font-size:11px;margin:0 0 5px;color:inherit;}
.ti-report-tech {margin-top:17px;border-top:1px solid var(--ti-edge);padding-top:14px;font-size:11px;color:var(--ti-muted);}
.ti-report-tech summary {cursor:pointer;font-weight:700;color:var(--ti-soft);padding:5px 0;}
.ti-report-tech p {margin:7px 0;overflow-wrap:anywhere;}
.ti-footer {border-top:1px solid var(--ti-edge);margin-top:28px;padding:24px 0;text-align:center;font-size:11px;line-height:1.8;color:var(--ti-muted);}
.ti-section-heading {margin:26px 0 22px;}.ti-section-heading h1 {font-size:27px;margin:10px 0 7px;letter-spacing:-.8px;color:var(--ti-text);}
.ti-flow {display:grid;gap:14px;color:var(--ti-soft);}
.ti-flow-lead,.ti-flow-empty,.ti-flow-outcome {border:1px solid var(--ti-edge);background:var(--ti-panel);border-radius:13px;padding:18px;}
.ti-flow-lead h3 {color:var(--ti-text);margin:8px 0;}.ti-flow-lead p {font-size:12px;color:var(--ti-muted);}
.ti-flow-eyebrow {color:var(--ti-cyan);font-size:10px;letter-spacing:1.5px;font-weight:800;}
.ti-flow-agents {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;}
.ti-flow-agent {border:1px solid var(--ti-edge);background:var(--ti-panel2);border-radius:11px;padding:17px;min-width:0;}
.ti-flow-agent strong {font-size:14px;color:var(--ti-text);}.ti-agent-class {font-size:10px;overflow-wrap:anywhere;color:var(--ti-cyan);margin:5px 0;}
.ti-flow-agent p {font-size:11px;line-height:1.7;color:var(--ti-muted);}
.ti-flow-status {display:inline-block;border:1px solid var(--ti-edge-strong);background:var(--ti-panel3);border-radius:20px;padding:5px 9px;font-size:10px;color:var(--ti-soft);}
.ti-qwen-service {border:1px dashed var(--ti-edge-strong);border-left:3px solid var(--ti-gold);padding:17px;border-radius:12px;background:var(--ti-panel3);}
.ti-qwen-service h4 {margin:0 0 8px;color:var(--ti-gold);}.ti-qwen-service p {font-size:12px;color:var(--ti-muted);}
.ti-flow-events {list-style:none;padding:0;margin:0;display:grid;gap:6px;}
.ti-flow-event {display:flex;align-items:center;gap:11px;border:1px solid var(--ti-edge);background:var(--ti-panel3);padding:10px 13px;border-radius:9px;flex-wrap:wrap;}
.ti-flow-event>div {flex:1;min-width:130px;}.ti-flow-event strong {font-size:12px;}
.ti-flow-sequence {display:grid;place-items:center;min-width:28px;min-height:28px;border-radius:7px;background:var(--ti-panel2);color:var(--ti-cyan);font-size:11px;font-weight:700;}
.ti-flow-event-task,.ti-flow-event-note,.ti-flow-caveat {font-size:11px;color:var(--ti-muted);}
.ti-trace-caption {color:var(--ti-muted);font-size:12px;}
.ti-table-wrap {min-width:0;overflow:auto;}
footer {background:transparent!important;}
@media(min-width:1000px) {#ti-input-pane {flex:0 1 38%!important;} #ti-result-pane {flex:1 1 58%!important;}}
@media(max-width:999px) {
 .gradio-container {padding:0 16px 20px!important;}
 #ti-analysis-row {flex-direction:column!important;} #ti-input-pane,#ti-result-pane {width:100%!important;flex:auto!important;}
 .ti-hero-tags {display:none;}.ti-flow-agents {grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media(max-width:600px) {
 .gradio-container {padding:0 12px 18px!important;}
 #ti-topbar {flex-direction:column;align-items:stretch;gap:12px;}
 #ti-nav [role="tablist"] {justify-content:space-between;gap:2px;overflow:auto;}
 #ti-nav [role="tab"] {padding:9px 11px!important;font-size:12px;}
 .ti-hero {margin:21px 0 20px;}.ti-hero h1 {font-size:27px;}
 .ti-pane {padding:17px!important;}.ti-report-grid {grid-template-columns:minmax(0,1fr);}
 .ti-flow-agents {grid-template-columns:minmax(0,1fr);}.ti-brand-name {font-size:18px;}
 .ti-placeholder {min-height:250px;padding:20px;}.ti-report-summary {padding:17px;}
}
@media(prefers-reduced-motion:reduce) {*,*::before,*::after {transition:none!important;scroll-behavior:auto!important;}}
"""
