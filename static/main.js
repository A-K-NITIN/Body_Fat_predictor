// ── Validation rules (min/max match dataset ranges)
const RULES = {
  Density: { min: 0.95,  max: 1.12,  label: 'Density'  },
  Age:     { min: 18,    max: 100,   label: 'Age'       },
  Weight:  { min: 80,    max: 420,   label: 'Weight'    },
  Height:  { min: 55,    max: 85,    label: 'Height'    },
  Neck:    { min: 28,    max: 60,    label: 'Neck'      },
  Chest:   { min: 70,    max: 150,   label: 'Chest'     },
  Abdomen: { min: 60,    max: 160,   label: 'Abdomen'   },
  Hip:     { min: 75,    max: 160,   label: 'Hip'       },
  Thigh:   { min: 40,    max: 100,   label: 'Thigh'     },
  Knee:    { min: 28,    max: 55,    label: 'Knee'      },
  Ankle:   { min: 16,    max: 38,    label: 'Ankle'     },
  Biceps:  { min: 22,    max: 50,    label: 'Biceps'    },
  Forearm: { min: 18,    max: 40,    label: 'Forearm'   },
  Wrist:   { min: 14,    max: 24,    label: 'Wrist'     },
};

// ── Validate a single field
function validateField(name, value) {
  const r = RULES[name];
  if (value === '' || value === null || isNaN(Number(value))) {
    return `${r.label} is required`;
  }
  const v = parseFloat(value);
  if (v < r.min || v > r.max) {
    return `Must be ${r.min} – ${r.max}`;
  }
  return null;
}

// ── Live inline validation on blur / input
Object.keys(RULES).forEach(name => {
  const input = document.getElementById(name);
  const errEl = document.getElementById(`err-${name}`);
  if (!input) return;

  input.addEventListener('blur', () => {
    const err = validateField(name, input.value);
    errEl.textContent = err || '';
    input.classList.toggle('is-invalid', !!err);
  });

  input.addEventListener('input', () => {
    if (!input.classList.contains('is-invalid')) return;
    const err = validateField(name, input.value);
    errEl.textContent = err || '';
    input.classList.toggle('is-invalid', !!err);
  });
});

// ── Validate whole form
function validateAll() {
  let valid = true;
  const payload = {};
  Object.keys(RULES).forEach(name => {
    const input = document.getElementById(name);
    const errEl = document.getElementById(`err-${name}`);
    const err = validateField(name, input.value);
    if (err) {
      errEl.textContent = err;
      input.classList.add('is-invalid');
      valid = false;
    } else {
      payload[name] = parseFloat(input.value);
    }
  });
  return { valid, payload };
}

// ── Move needle on horizontal BFP scale
// Scale: 2% → 50%, mapped 0% → 100% track width
function moveNeedle(pct) {
  const needle = document.getElementById('bfp-needle');
  const MIN = 2, MAX = 50;
  const clamped = Math.min(MAX, Math.max(MIN, pct));
  const pos = ((clamped - MIN) / (MAX - MIN)) * 100;
  // Slight delay so CSS transition is visible
  setTimeout(() => { needle.style.left = pos + '%'; }, 80);
}

// ── Count-up animation
function countUp(target, duration = 1000) {
  const el = document.getElementById('result-pct');
  const from = parseFloat(el.textContent) || 0;
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (target - from) * ease).toFixed(1);
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = target.toFixed(1);
  }
  requestAnimationFrame(step);
}

// ── Show result card
function showResult(data) {
  const card = document.getElementById('result-card');
  card.classList.remove('hidden');
  // Re-trigger animation
  card.style.animation = 'none';
  void card.offsetWidth;
  card.style.animation = '';

  document.getElementById('result-category').textContent = data.category;
  document.getElementById('result-desc').textContent = data.description;

  countUp(data.body_fat_percentage);
  moveNeedle(data.body_fat_percentage);

  setTimeout(() => {
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);
}

// ── Load model info into badge
async function loadInfo() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    document.getElementById('accuracy-badge').textContent =
      `CV Accuracy: ${info.mean_cv_accuracy}%  |  R2: ${info.r2_score}`;
  } catch {
    document.getElementById('accuracy-badge').textContent = 'Model ready';
  }
}

// ── Form submit
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const { valid, payload } = validateAll();

  if (!valid) {
    const form = document.getElementById('predict-form');
    form.classList.remove('shake');
    void form.offsetWidth;
    form.classList.add('shake');
    form.addEventListener('animationend', () => form.classList.remove('shake'), { once: true });
    document.querySelector('.is-invalid')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }

  const btn    = document.getElementById('submit-btn');
  const label  = document.getElementById('btn-label');
  const spin   = document.getElementById('btn-spinner');

  btn.disabled = true;
  label.textContent = 'Calculating…';
  spin.classList.remove('hidden');

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      alert('API error: ' + JSON.stringify(err.detail));
      return;
    }

    const data = await res.json();
    showResult(data);

  } catch (err) {
    alert('Cannot reach server. Make sure the API is running on port 8000.');
    console.error(err);
  } finally {
    btn.disabled = false;
    label.textContent = 'Estimate Body Fat';
    spin.classList.add('hidden');
  }
});

// ── Reset
document.getElementById('reset-btn').addEventListener('click', () => {
  document.getElementById('predict-form').reset();
  document.querySelectorAll('.is-invalid').forEach(el => el.classList.remove('is-invalid'));
  document.querySelectorAll('.field-error').forEach(el => el.textContent = '');
  document.getElementById('result-card').classList.add('hidden');
  document.getElementById('bfp-needle').style.left = '0%';
});

// ── Init
loadInfo();
