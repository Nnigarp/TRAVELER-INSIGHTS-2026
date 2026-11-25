"""Shared presentation for UTB Claro Premium and Caribe Nocturno."""
import gradio as gr

_PRIMARY = gr.themes.colors.teal
THEME = gr.themes.Soft(primary_hue=_PRIMARY, secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate, font=["Manrope", "DM Sans", "ui-sans-serif", "system-ui", "sans-serif"]).set(
    body_background_fill="#f3f7fc", body_background_fill_dark="#061217",
    block_background_fill="#ffffff", block_background_fill_dark="#0c1f27",
    body_text_color="#102b46", body_text_color_dark="#eefaf9",
    background_fill_primary="#ffffff", background_fill_primary_dark="#0c1f27",
    background_fill_secondary="#eef4f8", background_fill_secondary_dark="#112a35")

THEME_CHOICES = ("UTB Claro Premium", "Caribe Nocturno")

# The preference is local to this origin. All analysis state stays in Gradio.
_APPLY_THEME = """
 const dark = value === 'Caribe Nocturno';
 const root = document.documentElement;
 root.lang = 'es';
 root.dataset.tiTheme = dark ? 'caribe' : 'utb';
 root.classList.toggle('dark', dark);
 document.querySelectorAll('gradio-app, .gradio-container').forEach(
   element => element.classList.toggle('dark', dark)
 );
"""
THEME_JS = """(value) => {""" + _APPLY_THEME + """
 try { localStorage.setItem('traveler-insights:theme', value); } catch (_) {}
 return [];
}"""
INITIAL_THEME_JS = """() => {
 if (!document.documentElement.dataset.tiSubmitGuard) {
   document.documentElement.dataset.tiSubmitGuard = '1';
   document.addEventListener('click', event => {
     const state = document.documentElement.dataset;
     if (event.target.closest?.('#ti-cancel')) delete state.tiPending;
     if (!event.target.closest?.('#ti-analyze')) return;
     if (state.tiPending) {
       event.preventDefault();
       event.stopImmediatePropagation();
     } else {
       state.tiPending = '1';
     }
   }, true);
 }
 let value = 'UTB Claro Premium';
 try {
   if (localStorage.getItem('traveler-insights:theme') === 'Caribe Nocturno') {
     value = 'Caribe Nocturno';
   }
 } catch (_) {}
""" + _APPLY_THEME + """
 return [value];
}"""

UI_CSS = """
:root, :root[data-ti-theme="utb"] {
 --ti-bg:#f3f7fc; --ti-panel:#fff; --ti-panel2:#eaf2fb; --ti-panel3:#f6f9fd;
 --ti-edge:#d7e3f0; --ti-edge-strong:#a6bed4; --ti-muted:#49657d; --ti-text:#102b46;
 --ti-soft:#27435e; --ti-primary:#0b4f8a; --ti-cyan:#08766b; --ti-gold:#894008;
 --ti-green:#047454; --ti-red:#ad1740; --ti-gold-bg:#fff3d6; --ti-gold-edge:#dca639;
 --ti-shadow:0 4px 10px #17375308,0 16px 40px #1737530c; --ti-summary:#e8f4f8; --ti-focus:#0b4f8a;
 --ti-lift:#dceafa; --ti-hero:linear-gradient(108deg,#fff 12%,#e9f3fb 100%);
 color-scheme:light;
}
:root[data-ti-theme="caribe"] {
 --ti-bg:#061217; --ti-panel:#0c1f27; --ti-panel2:#112a35; --ti-panel3:#09181f;
 --ti-edge:#1b3f4e; --ti-edge-strong:#386375; --ti-muted:#a6c4cf; --ti-text:#eefaf9;
 --ti-soft:#d3e8eb; --ti-primary:#0f766e; --ti-cyan:#5eead4; --ti-gold:#fcd34d;
 --ti-green:#6ee7b7; --ti-red:#fda4af; --ti-gold-bg:#2c2514; --ti-gold-edge:#796025;
 --ti-shadow:0 4px 24px #0003; --ti-summary:#0f2731; --ti-focus:#5eead4;
 --ti-lift:#193947; --ti-hero:linear-gradient(108deg,#0c1f27 12%,#10303a 100%);
 color-scheme:dark;
}
/* The canvas belongs to the viewport. Gradio's host must paint the same theme
   even when it sits outside the Blocks container or applies an inline fill. */
html, body, gradio-app {
 background:var(--ti-bg)!important;
 color:var(--ti-text)!important;
 width:100%;
}
.gradio-container {
 box-sizing:border-box;
 width:100%!important;
 max-width:none!important;
 margin:0 auto!important;
 background:var(--ti-bg)!important;
 color:var(--ti-text)!important;
 font-family:Manrope,'DM Sans',ui-sans-serif,system-ui,sans-serif!important;
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

 --body-text-size:16px; --text-md:1rem; --text-sm:.875rem;
 --input-text-size:1rem; --block-label-text-size:.875rem;
 --accordion-text-color:var(--ti-text); --accordion-text-color-dark:var(--ti-text);
 --checkbox-label-background-fill-hover:var(--ti-lift);
 --checkbox-label-background-fill-hover-dark:var(--ti-lift);
 padding:0 clamp(14px,2.4vw,46px) 24px!important;
 min-height:100vh;
}
/* Gradio also pads its inner app shell. Keep a single responsive gutter. */
.gradio-container > .main {padding:0!important;margin:0 auto!important;max-width:none!important;width:100%!important;min-width:0;}
.gradio-container a:focus-visible, .gradio-container button:focus-visible, .gradio-container input:focus-visible,
.gradio-container textarea:focus-visible, .gradio-container summary:focus-visible,
.gradio-container [role="tab"]:focus-visible {
 outline:3px solid var(--ti-focus)!important; outline-offset:3px!important;
}
.gradio-container button {transition:background .18s,border-color .18s;min-height:44px;}
.gradio-container button.secondary {background:var(--ti-panel2)!important;color:var(--ti-soft)!important;
 border:1px solid var(--ti-edge-strong)!important;box-shadow:none!important;}
.gradio-container button.secondary:hover {background:var(--ti-lift)!important;}
.gradio-container input[type="checkbox"],.gradio-container input[type="radio"] {border:1px solid var(--ti-muted)!important;}
.gradio-container textarea {line-height:1.7!important;}
.gradio-container label, .gradio-container .prose {color:var(--ti-soft);}
.gradio-container .prose p, .gradio-container .prose li {color:inherit;}
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3,
.gradio-container .prose h4 {color:var(--ti-text);}
#ti-topbar {align-items:center;border-bottom:1px solid var(--ti-edge);padding:24px 0 18px;gap:18px;}
.ti-brand {display:flex;gap:12px;align-items:center;}
.ti-brand-mark {display:grid;place-items:center;background:linear-gradient(140deg,#0b4f8a,#08766b);
 width:48px;height:48px;border-radius:14px;color:#fff!important;flex:none;box-shadow:0 8px 24px #14b8a629;}
.ti-brand svg {width:27px;height:27px;color:#fff;}
.ti-brand-name {font-size:clamp(20px,2vw,26px);font-weight:800;letter-spacing:-.7px;color:var(--ti-text);}
.ti-brand-name span {color:var(--ti-cyan);}
.ti-brand-sub {font-size:14px;color:var(--ti-muted);letter-spacing:.08em;font-weight:700;margin-top:4px;}
#ti-theme {background:transparent!important;border:none!important;padding:0;}
#ti-topbar .form {background:transparent!important;border:none!important;box-shadow:none!important;}
#ti-theme label {font-size:14px!important;min-height:44px;}
#ti-nav {border:none;}
#ti-nav [role="tablist"] {display:flex;gap:6px;border-bottom:1px solid var(--ti-edge);padding:6px 0;}
#ti-nav [role="tab"], #ti-nav .tab-container button {border:1px solid transparent;border-radius:10px!important;font-weight:700;
 color:var(--ti-muted);padding:9px 18px!important;background:transparent;min-height:44px;font-size:15px;}
#ti-nav [role="tab"][aria-selected="true"] {background:var(--ti-primary)!important;color:#fff!important;border-color:var(--ti-primary)!important;}
#ti-nav > .tabitem {border:none;padding:0;min-width:0;}
#ti-nav [role="tab"]:hover {background:var(--ti-panel2);color:var(--ti-text);}
/* Compact introduction, then the task. No technical availability badges. */
.ti-hero,.ti-section-heading {
 margin:clamp(16px,2vw,28px) 0 20px;
 padding:clamp(18px,3vw,36px);
 border:1px solid var(--ti-edge);
 border-inline-start:4px solid var(--ti-primary);
 border-radius:20px;
 background:var(--ti-hero);
}
.ti-eyebrow {font-size:14px;letter-spacing:.11em;font-weight:800;color:var(--ti-cyan);text-transform:uppercase;}
.gradio-container .ti-hero h1,.gradio-container .ti-section-heading h1 {
 font-size:clamp(26px,2.5vw,42px);font-weight:800;line-height:1.15;letter-spacing:-.035em;margin:9px 0;
}
.ti-hero p,.ti-copy {color:var(--ti-muted);font-size:16px;line-height:1.6;margin:0;max-width:74ch;}
#ti-analysis-row {display:grid!important;grid-template-columns:minmax(0,1fr);align-items:start;gap:clamp(16px,2vw,30px);}
#ti-input-pane,#ti-result-pane {min-width:0!important;width:100%!important;}
#ti-result-pane {container-type:inline-size;}
.ti-pane {border:1px solid var(--ti-edge)!important;border-radius:18px!important;background:var(--ti-panel)!important;
 box-shadow:var(--ti-shadow);overflow:visible;padding:clamp(16px,1.65vw,28px)!important;min-width:0;}
.ti-pane-head {border-bottom:1px solid var(--ti-edge);padding-bottom:12px;margin-bottom:4px;}
.gradio-container .ti-pane-head h2 {font-size:22px;letter-spacing:-.025em;margin:0 0 5px;font-weight:800;color:var(--ti-text);}
.ti-pane-head p {font-size:15px;line-height:1.55;color:var(--ti-muted);margin:0;}
#ti-input-pane .block {box-shadow:none;}
#ti-qwen .info-text {color:var(--ti-muted)!important;font-size:14px;}
#ti-input-pane #ti-qwen {background:var(--ti-panel2);border:1px solid var(--ti-edge);border-radius:10px;padding:12px;}
#ti-char-count {text-align:right;font-size:14px;color:var(--ti-muted);min-height:20px;}
.ti-example-note {color:var(--ti-muted);font-size:14px;margin:8px 0 2px;}
.ti-example-row {gap:6px!important;}
.ti-example-row button {font-size:14px!important;padding:6px 9px!important;min-width:0!important;}
.ti-input-note {padding:12px 14px;border:1px solid var(--ti-edge);border-radius:10px;background:var(--ti-panel3);font-size:14px;line-height:1.7;color:var(--ti-muted);}
.ti-input-note strong {color:var(--ti-soft);}
.ti-placeholder {min-height:280px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;padding:32px;color:var(--ti-muted);gap:12px;}
.ti-placeholder svg {width:46px;height:46px;color:var(--ti-cyan);}
.ti-placeholder h3 {font-size:17px;color:var(--ti-text);margin:0;}
.ti-placeholder p {font-size:14px;line-height:1.75;max-width:42ch;margin:0;}
.ti-report {color:var(--ti-soft);font-size:16px;line-height:1.6;overflow-wrap:anywhere;}
.ti-report h3,.ti-report h4,.ti-report p {margin:0;}
.ti-report-source {font-size:14px;color:var(--ti-muted);margin-bottom:14px!important;}
.ti-report-summary {border-inline-start:4px solid var(--ti-primary)!important;background:radial-gradient(ellipse at 100% 0%,#14b8a615,transparent 65%),var(--ti-summary);
 border:1px solid var(--ti-edge-strong);padding:19px 21px;border-radius:12px;margin-bottom:16px;}
.ti-report-summary h4 {font-size:13px;letter-spacing:1.6px;text-transform:uppercase;color:var(--ti-cyan);margin-bottom:10px;}
.ti-report-summary p {font-size:clamp(17px,1.2vw,21px);font-weight:650;line-height:1.5;color:var(--ti-text);}
.ti-indicators {display:flex;flex-wrap:wrap;gap:7px;margin-bottom:17px;}
.ti-indicator {font-size:13px;color:var(--ti-muted);border:1px solid var(--ti-edge);border-radius:8px;background:var(--ti-panel2);padding:8px 10px;}
.ti-indicator strong {display:block;color:var(--ti-text);font-size:15px;}
.ti-sent-negativo strong {color:var(--ti-red);}.ti-sent-positivo strong {color:var(--ti-green);}
.ti-report-grid {display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;}
.ti-report-block {border:1px solid var(--ti-edge);border-top:3px solid var(--ti-cyan);background:var(--ti-panel3);
 border-radius:11px;padding:16px 17px;min-width:0;overflow-wrap:anywhere;}
.ti-report-block h4 {font-size:13px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px;font-weight:800;color:var(--ti-cyan);}
.ti-report-problem {border-top-color:var(--ti-red);}.ti-report-problem h4 {color:var(--ti-red);}
.ti-report-rec {border-top-color:var(--ti-green);}.ti-report-rec h4 {color:var(--ti-green);}
.ti-report-action {border-top-color:var(--ti-gold);}.ti-report-action h4 {color:var(--ti-gold);}
.ti-report blockquote {margin:0 0 9px;border-left:2px solid var(--ti-cyan);padding-left:12px;font-size:14px;font-style:italic;line-height:1.7;}
.ti-warnings {margin-top:15px;border:1px solid var(--ti-gold-edge);background:var(--ti-gold-bg);border-radius:10px;padding:12px 15px;color:var(--ti-gold);font-size:14px;line-height:1.7;}
.ti-warnings h4 {font-size:14px;margin:0 0 5px;color:inherit;}
.ti-report-tech {margin-top:17px;border-top:1px solid var(--ti-edge);padding-top:14px;font-size:14px;color:var(--ti-muted);}
.ti-report-tech summary {cursor:pointer;font-weight:700;color:var(--ti-soft);padding:5px 0;}
.ti-report-tech p {margin:7px 0;overflow-wrap:anywhere;}
.ti-footer {border-top:1px solid var(--ti-edge);margin-top:28px;padding:24px 0;text-align:center;font-size:14px;line-height:1.8;color:var(--ti-muted);}

.ti-flow {display:grid;gap:14px;color:var(--ti-soft);}
.ti-flow-lead,.ti-flow-empty,.ti-flow-outcome {border:1px solid var(--ti-edge);background:var(--ti-panel);border-radius:13px;padding:18px;}
.ti-flow-lead h3 {color:var(--ti-text);margin:8px 0;}.ti-flow-lead p {font-size:14px;color:var(--ti-muted);}
.ti-flow-eyebrow {color:var(--ti-cyan);font-size:13px;letter-spacing:1.5px;font-weight:800;}
.ti-flow-agents {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;}
.ti-flow-agent {border:1px solid var(--ti-edge);background:var(--ti-panel2);border-radius:11px;padding:17px;min-width:0;}
.ti-flow-agent strong {font-size:14px;color:var(--ti-text);}.ti-agent-class {font-size:13px;overflow-wrap:anywhere;color:var(--ti-cyan);margin:5px 0;}
.ti-flow-agent p {font-size:14px;line-height:1.7;color:var(--ti-muted);}
.ti-flow-status {display:inline-block;border:1px solid var(--ti-edge-strong);background:var(--ti-panel3);border-radius:20px;padding:5px 9px;font-size:13px;color:var(--ti-soft);}
.ti-qwen-service {border:1px dashed var(--ti-edge-strong);border-left:3px solid var(--ti-gold);padding:17px;border-radius:12px;background:var(--ti-panel3);}
.ti-qwen-service h4 {margin:0 0 8px;color:var(--ti-gold);}.ti-qwen-service p {font-size:14px;color:var(--ti-muted);}
.ti-flow-events {list-style:none;padding:0;margin:0;display:grid;gap:6px;}
.ti-flow-event {display:flex;align-items:center;gap:11px;border:1px solid var(--ti-edge);background:var(--ti-panel3);padding:10px 13px;border-radius:9px;flex-wrap:wrap;}
.ti-flow-event>div {flex:1;min-width:130px;}.ti-flow-event strong {font-size:14px;}
.ti-flow-sequence {display:grid;place-items:center;min-width:28px;min-height:28px;border-radius:7px;background:var(--ti-panel2);color:var(--ti-cyan);font-size:14px;font-weight:700;}
.ti-flow-event-task,.ti-flow-event-note,.ti-flow-caveat {font-size:14px;color:var(--ti-muted);}
.ti-trace-caption {color:var(--ti-muted);font-size:14px;}
/* Scroll belongs to the native table or editor, not the whole accordion. */
.ti-table-wrap, .ti-technical, .ti-technical .block {min-width:0!important;max-width:100%!important;}
.ti-technical, .ti-technical .cm-editor {overflow:hidden!important;}
.ti-technical .cm-scroller {overflow:auto!important;overscroll-behavior:contain;}
.ti-technical button {color:var(--ti-text)!important;}
.ti-technical .cm-content {overflow-wrap:anywhere;}
.ti-table-wrap [data-testid="dataframe"] {min-width:0;max-width:100%;}
.ti-config-context {margin:0 0 14px;color:var(--ti-muted);font-size:14px;}
.ti-config-context strong {color:var(--ti-text);}
#ti-audio-config,#ti-batch-config {align-items:center;gap:8px 16px;margin-bottom:8px;}
.ti-run-status .prose {color:var(--ti-text);}
.ti-pane [data-testid="status-tracker"]:not(.hide) {
 color:var(--ti-text);
 font-size:15px;
}
#ti-batch-table [role="grid"] {max-width:100%;}
#ti-batch-table [role="columnheader"] {font-size:14px;}
#ti-batch-table .virtual-table-viewport {overscroll-behavior-inline:contain;}
.ti-result-empty {border:1px dashed var(--ti-edge-strong);border-radius:14px;background:var(--ti-panel3);}
.gradio-container p[role="alert"] {padding:16px;border:1px solid var(--ti-red);border-radius:12px;color:var(--ti-red);}
.ti-flow-order {display:flex;flex-wrap:wrap;gap:8px 22px;list-style:none;margin:0;padding:0;}
.ti-flow-order li {position:relative;padding:8px 12px;border:1px solid var(--ti-edge);border-radius:8px;background:var(--ti-panel2);font-size:14px;}
.ti-flow-order li + li::before {content:"→";position:absolute;inset-inline-start:-20px;color:var(--ti-cyan);}
.ti-flow-order .ti-flow-optional {border-style:dashed;}
.ti-flow-outcome {border-inline-start:4px solid var(--ti-cyan);}
footer {background:transparent!important;}
@media(min-width:1000px) {
 #ti-analysis-row {grid-template-columns:minmax(320px,2fr) minmax(0,3fr);}
}
@media(min-width:1400px) {
 #ti-analysis-row {grid-template-columns:minmax(380px,1fr) minmax(0,1.75fr);}
}
@container(max-width:540px) {
 .ti-report-grid {grid-template-columns:minmax(0,1fr);}
}
@media(max-width:999px) {
 .ti-flow-agents {grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media(max-width:600px) {
 #ti-topbar {flex-direction:column;align-items:stretch;gap:10px;}
 #ti-topbar > .column {min-width:0!important;width:100%;}
 #ti-nav [role="tablist"] {justify-content:space-between;gap:2px;overflow:auto;}
 #ti-nav [role="tab"], #ti-nav .tab-container button {padding:8px 8px!important;font-size:14px;}
 .ti-report-grid,.ti-flow-agents {grid-template-columns:minmax(0,1fr);}
 .ti-placeholder {min-height:200px;padding:18px;}
 .ti-report-summary {padding:16px;}
}
@media(prefers-reduced-motion:reduce) {*,*::before,*::after {transition:none!important;scroll-behavior:auto!important;}}

#ti-topbar {padding:12px 0;}
.ti-hero {text-align:center;background:transparent;border:0;padding:12px;margin:18px auto;}
.ti-hero p {margin:auto;}
#ti-composer {border:1px solid var(--ti-edge-strong);background:var(--ti-panel);border-radius:16px;padding:12px;min-width:0;}
#ti-composer #ti-review textarea, #ti-composer #ti-review textarea:is(:focus,:hover,:focus-visible,:active,:disabled) {
 background:var(--ti-panel)!important;color:var(--ti-text)!important;-webkit-text-fill-color:var(--ti-text)!important;
 caret-color:var(--ti-text)!important;opacity:1!important;font-size:17px!important;line-height:1.6!important;
 padding:14px!important;border:1px solid var(--ti-edge-strong)!important;border-radius:10px!important;}
#ti-review textarea::placeholder {color:var(--ti-muted)!important;-webkit-text-fill-color:var(--ti-muted)!important;opacity:1!important;}
#ti-review textarea::selection {background:var(--ti-primary)!important;color:white!important;-webkit-text-fill-color:white!important;}
#ti-review textarea:-webkit-autofill {box-shadow:0 0 0 1000px var(--ti-panel) inset!important;}
#ti-tools {align-items:center;gap:6px;flex-wrap:wrap;}
#ti-tools #ti-qwen {background:transparent;border:none;}
#ti-input-pane :is(.form,.block,.column,.wrap,.label-wrap),#ti-audio-panel,#ti-language-panel {min-width:0!important;box-sizing:border-box;max-width:100%;}
#ti-audio-panel audio {width:100%;max-width:100%;}
#ti-input-pane .label-wrap {white-space:normal;}
.ti-operations {color:var(--ti-text);padding:clamp(14px,2vw,28px);border:1px solid var(--ti-edge);background:var(--ti-panel);border-radius:16px;}
.ti-operations h2 {font-size:22px;margin:0 0 16px;}
.ti-operations dl {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(145px,100%),1fr));gap:14px;margin:0;}
.ti-operations dt,.ti-runtime-map span,.ti-coordinator span {font-size:14px;color:var(--ti-muted);}
.ti-operations dd {margin:3px 0 0;font-weight:650;overflow-wrap:anywhere;}
.ti-decision {border-inline-start:3px solid var(--ti-cyan);padding:8px 12px;background:var(--ti-panel2);}
.ti-coordinator {margin:22px 0 10px;border-bottom:2px solid var(--ti-primary);display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding-bottom:8px;}
.ti-runtime-map {list-style:none!important;padding:0!important;display:flex;gap:4px;margin:10px 0 18px;}
.ti-runtime-map li {min-width:0;flex:1;display:grid;align-content:start;gap:5px;border-top:3px solid var(--ti-cyan);background:var(--ti-panel2);padding:10px;overflow-wrap:anywhere;}
.ti-runtime-map .ti-node-idle {border-top:3px dashed var(--ti-edge-strong);}
.ti-runtime-map .ti-node-error {border-color:var(--ti-red);}
.ti-operations summary,.ti-timeline summary {cursor:pointer;color:var(--ti-text);font-weight:650;padding:12px 0;}
@media(max-width:800px){.ti-runtime-map {flex-direction:column;} .ti-runtime-map li {display:flex;justify-content:space-between;border-top:0;border-inline-start:3px solid var(--ti-cyan);}}
#ti-analysis-row {grid-template-columns:minmax(0,1fr);width:min(100%,1024px);margin-inline:auto;}
#ti-input-pane {box-shadow:none;border:0!important;background:transparent!important;padding:0!important;}
#ti-composer {background:var(--ti-panel)!important;padding:clamp(14px,2vw,24px)!important;gap:8px;}
#ti-input-pane .ti-example-row {max-width:650px;}
#ti-tools .form {flex:1 1 270px!important;min-width:0!important;}
#ti-tools > button {flex:0 1 auto!important;}
#ti-tools .block {flex:1 1 120px!important;}
#ti-composer .form,#ti-composer .block {border:0!important;box-shadow:none!important;background:transparent!important;}
#ti-char-count {line-height:1.3;}
#ti-result .ti-placeholder {min-height:110px;}
@media(min-width:801px){.ti-operations dl {grid-template-columns:repeat(4,minmax(0,1fr));}
.ti-runtime-map li:not(:last-child)::after {content:"→";position:absolute;right:-8px;top:8px;color:var(--ti-muted);z-index:1;}
.ti-runtime-map {gap:12px;}.ti-runtime-map li {position:relative;}}
.ti-runtime-map .ti-node-review {border-color:var(--ti-gold);background:var(--ti-gold-bg);}
#ti-dictate::before {content:"🎙";font-family:"Segoe UI Symbol",sans-serif;margin-inline-end:4px;}
"""
