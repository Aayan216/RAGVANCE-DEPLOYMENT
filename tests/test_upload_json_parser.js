/* Tests for the safe JSON parsing in frontend/templates/upload.html.
 *
 * Extracts the page's <script> block, runs it in a vm sandbox with stubbed
 * DOM/bootstrap/fetch/location, and verifies every response shape:
 * empty bodies, proxy HTML pages, JSON errors, success, legacy payloads,
 * delete flows — including that "Unexpected end of JSON input" never leaks.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const HTML = fs.readFileSync(
    path.join(__dirname, '..', 'frontend', 'templates', 'upload.html'),
    'utf8');

let PASSED = 0;
let FAILED = 0;

function ok(label, cond, extra) {
    if (cond) {
        PASSED += 1;
        console.log('PASS - ' + label);
    } else {
        FAILED += 1;
        console.log('FAIL - ' + label + (extra !== undefined ? ' :: ' + JSON.stringify(extra) : ''));
    }
}

const scriptMatch = HTML.match(/<script>([\s\S]*)<\/script>/);
ok('upload.html has a script block', !!scriptMatch);

let script = scriptMatch[1];
ok('script has exactly one JSON.parse (parser is the single parse point)',
    (script.match(/JSON\.parse/g) || []).length === 1,
    (script.match(/JSON\.parse/g) || []).length);
script = script.replace(/{% url 'process_document' 0 %}/g, '/process/0');
ok('django template url tag stripped for sandbox',
    !script.includes('{%'), script.slice(0, 80));

// ---------- DOM / framework stubs ----------
const toasts = [];            // {kind, msg} per showToast call
const reloads = { count: 0 };
let hiddenCount = 0;
let rowRemovals = 0;
const btns = {};              // process buttons by doc id
const deleteState = { modalShown: 0 };

function makeEl(id) {
    return {
        id: id || '',
        disabled: false,
        innerHTML: '',
        textContent: '',
        value: 'csrf-token',
        addEventListener(type, fn) { this['on_' + type] = fn; },
        remove() { rowRemovals += 1; },
    };
}

const docEls = {};            // getElementById cache
function getEl(id) {
    if (!docEls[id]) docEls[id] = makeEl(id);
    return docEls[id];
}

class Toast {
    constructor(el) { this.el = el; }
    show() {
        const isOk = this.el.id === 'successToast';
        toasts.push({
            kind: isOk ? 'success' : 'error',
            msg: getEl(isOk ? 'successToastMsg' : 'errorToastMsg').textContent,
        });
    }
}
class Modal {
    constructor(el) { this.el = el; Modal._registry.set(el, this); }
    show() { deleteState.modalShown += 1; }
    hide() { hiddenCount += 1; }
    static getInstance(el) { return Modal._registry.get(el); }
}
Modal._registry = new Map();

function processBtn(docId) {
    const key = 'proc-' + docId;
    if (!btns[key]) btns[key] = makeEl(key);
    return btns[key];
}

const documentStub = {
    getElementById(id) {
        if (id === 'confirmDeleteBtn') return getEl('confirmDeleteBtn');
        return getEl(id);
    },
    querySelector(sel) {
        const m = sel.match(/^\[data-doc-id="(\d+)"\]$/);
        if (m) return processBtn(m[1]);
        if (sel === '[name=csrfmiddlewaretoken]') return getEl('csrf');
        return getEl(sel);
    },
};

// fetch is swapped per scenario (global lookup at call time).
let fetchImpl = async () => { throw new Error('fetch not stubbed'); };

const sandbox = {
    document: documentStub,
    bootstrap: { Toast, Modal },
    location: { reload() { reloads.count += 1; } },
    fetch: (...args) => fetchImpl(...args),
    console,
    Error,
    JSON,
    Promise,
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);

function resp(status, body) {
    return { ok: status >= 200 && status < 300, status, text: async () => body };
}

async function runProcess(scenario) {
    toasts.length = 0;
    reloads.count = 0;
    fetchImpl = scenario.fetch;
    const btn = processBtn(42);
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-gear"></i> Process';
    await sandbox.processDocument(42);
    return { btn, toast: toasts[0] };
}

async function runDelete(scenario) {
    toasts.length = 0;
    reloads.count = 0;
    hiddenCount = 0;
    fetchImpl = scenario.fetch;
    sandbox.confirmDelete(42, 'fixture doc');
    const btn = getEl('confirmDeleteBtn');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-trash"></i> Delete Permanently';
    await btn.on_click();
    return { btn, toast: toasts[0] };
}

(async () => {
    // 1. empty body + HTTP 200
    let r = await runProcess({ fetch: async () => resp(200, '') });
    ok('empty 200 -> error toast mentions HTTP 200',
        r.toast && r.toast.kind === 'error'
        && r.toast.msg.includes('empty response (HTTP 200)'), r.toast);
    ok('empty 200 -> never shows raw SyntaxError',
        r.toast && !r.toast.msg.includes('Unexpected end of JSON input'), r.toast);
    ok('empty 200 -> no reload', reloads.count === 0, reloads.count);
    ok('empty 200 -> button restored',
        r.btn.disabled === false && r.btn.innerHTML.includes('Process'), r.btn);

    // 2. empty body + proxy 502
    r = await runProcess({ fetch: async () => resp(502, '') });
    ok('empty 502 -> "empty response (HTTP 502)"',
        r.toast && r.toast.msg.includes('empty response (HTTP 502)'), r.toast);

    // 3. proxy HTML page (non-JSON) + 502
    r = await runProcess({ fetch: async () => resp(502, '<html><body>502 Bad Gateway</body></html>') });
    ok('HTML 502 -> "invalid response (HTTP 502)"',
        r.toast && r.toast.msg.includes('invalid response (HTTP 502)'), r.toast);
    ok('HTML 502 -> never raw SyntaxError',
        r.toast && !r.toast.msg.includes('Unexpected end of JSON input'), r.toast);

    // 4. JSON 400 with backend error text (chunk ceiling)
    const ceilingMsg = 'Document contains too much content to process safely.';
    r = await runProcess({
        fetch: async () => resp(400, JSON.stringify({
            success: false, status: 'error', error: ceilingMsg,
        })),
    });
    ok('JSON 400 -> backend message shown verbatim',
        r.toast && r.toast.msg === ceilingMsg, r.toast);
    ok('JSON 400 -> no reload', reloads.count === 0);

    // 5. JSON 500 handled error
    r = await runProcess({
        fetch: async () => resp(500, JSON.stringify({
            success: false, error: 'Document processing failed: corrupted file.',
        })),
    });
    ok('JSON 500 -> sanitized backend message shown',
        r.toast && r.toast.msg === 'Document processing failed: corrupted file.', r.toast);

    // 6. valid success -> reload
    r = await runProcess({
        fetch: async () => resp(200, JSON.stringify({
            success: true, status: 'success', chunks: 12, message: 'ok',
        })),
    });
    ok('success -> location.reload called', reloads.count === 1, reloads.count);
    ok('success -> no toast', toasts.length === 0, toasts);

    // 7. legacy already_processed payload (no success field)
    r = await runProcess({
        fetch: async () => resp(200, JSON.stringify({
            status: 'already_processed', chunks: 3, message: 'already',
        })),
    });
    ok('legacy already_processed -> reload (no success field needed)',
        reloads.count === 1, reloads.count);

    // 8. HTTP 200 but success:false JSON (defensive)
    r = await runProcess({
        fetch: async () => resp(200, JSON.stringify({
            success: false, message: 'not allowed right now',
        })),
    });
    ok('200 + success:false -> error toast, no reload',
        r.toast && r.toast.kind === 'error'
        && r.toast.msg === 'not allowed right now' && reloads.count === 0, r.toast);

    // 9. delete: network failure (fetch rejects) -> modal stays open
    let d = await runDelete({ fetch: async () => { throw new Error('offline'); } });
    ok('delete network failure -> toast has fallback text',
        d.toast && d.toast.kind === 'error', d.toast);
    ok('delete network failure -> modal NOT hidden (no JSON received)',
        hiddenCount === 0, hiddenCount);
    ok('delete network failure -> button restored',
        d.btn.disabled === false && d.btn.innerHTML.includes('Delete Permanently'));

    // 10. delete: JSON 500 error -> modal hidden (JSON was received)
    d = await runDelete({
        fetch: async () => resp(500, JSON.stringify({
            success: false, error: 'Database unavailable.',
        })),
    });
    ok('delete JSON 500 -> modal hidden (err.data set)',
        hiddenCount === 1, hiddenCount);
    ok('delete JSON 500 -> backend error shown',
        d.toast && d.toast.msg === 'Database unavailable.', d.toast);
    ok('delete JSON 500 -> no row removed', rowRemovals === 0, rowRemovals);
    ok('delete JSON 500 -> button restored',
        d.btn.disabled === false && d.btn.innerHTML.includes('Delete Permanently'));

    // 11. delete: success -> modal hidden, row removed, success toast
    const rowsBefore = rowRemovals;
    d = await runDelete({
        fetch: async () => resp(200, JSON.stringify({
            status: 'success', message: 'Document deleted.',
        })),
    });
    ok('delete success -> modal hidden', hiddenCount === 1, hiddenCount);
    ok('delete success -> row removed', rowRemovals === rowsBefore + 1, rowRemovals);
    ok('delete success -> success toast with backend message',
        d.toast && d.toast.kind === 'success'
        && d.toast.msg === 'Document deleted.', d.toast);
    ok('delete success -> button restored',
        d.btn.disabled === false && d.btn.innerHTML.includes('Delete Permanently'));

    // 12. delete: empty body -> error toast, modal hidden (JSON attempted)
    d = await runDelete({ fetch: async () => resp(200, '   ') });
    ok('delete empty body -> error toast mentions empty response',
        d.toast && d.toast.msg.includes('empty response'), d.toast);
    ok('delete empty body -> never raw SyntaxError',
        d.toast && !d.toast.msg.includes('Unexpected end of JSON input'), d.toast);

    console.log(`\n${PASSED}/${PASSED + FAILED} passed`);
    process.exit(FAILED ? 1 : 0);
})().catch((err) => {
    console.error(err);
    console.log(`\n${PASSED}/${PASSED + FAILED} passed (script crashed)`);
    process.exit(1);
});
