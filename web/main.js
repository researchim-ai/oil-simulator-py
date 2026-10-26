(function () {
  const caps = {
    "assets/wells_oil.png": "Дебит нефти, м³/сут. Голубая — Eclipse, пунктир — наш солвер, точки — история.",
    "assets/wells_water.png": "Дебит воды, м³/сут. Вода приходит позже нефти: к концу истории её на 0,6% больше, чем у Eclipse.",
    "assets/wells_gas.png": "Дебит газа, тыс. м³/сут. Свободный газ плюс растворённый.",
    "assets/wells_bhp.png": "Забойное давление, бар. Нули Eclipse — закрытая скважина. Участки на 1 атм — нижний предел забоя из deck, Eclipse упирается в него же.",
    "assets/map_so.png": "Нефтенасыщенность, среднее по разрезу. Южный сегмент светлеет к 2006 году.",
    "assets/map_sw.png": "Водонасыщенность. Закачка и подъём воды читаются как потемнение юга и центра.",
    "assets/map_sg.png": "Газонасыщенность. Свободный газ держится шапкой, в основном на севере.",
    "assets/map_p.png": "Давление, бар, среднее по разрезу с весом порового объёма."
  };

  document.querySelectorAll("[data-switch]").forEach(function (group) {
    const kind = group.getAttribute("data-switch");
    group.querySelectorAll("button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        group.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
        btn.setAttribute("aria-pressed", "true");
        if (kind === "well" || kind === "map") {
          const img = document.getElementById(kind === "well" ? "well" : "map");
          const cap = document.getElementById(kind === "well" ? "well-cap" : "map-cap");
          img.src = btn.dataset.src;
          cap.textContent = caps[btn.dataset.src];
        } else {
          paint();
        }
      });
    });
  });

  const stage = document.getElementById("stage");
  let phase = "so";
  let when = "end";
  function paint() {
    if (!stage) return;
    phase = document.querySelector("[data-switch=phase] button[aria-pressed=true]").dataset.phase;
    when = document.querySelector("[data-switch=when] button[aria-pressed=true]").dataset.when;
    stage.src = "assets/3d_" + phase + "_" + when + ".png";
  }
  if (stage) ["so", "sw", "sg"].forEach(function (p) {
    ["start", "end"].forEach(function (w) {
      const im = new Image();
      im.src = "assets/3d_" + p + "_" + w + ".png";
    });
  });

  const links = Array.from(document.querySelectorAll(".rail a"));
  const sections = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
  function mark() {
    const y = window.scrollY + 120;
    let current = 0;
    sections.forEach(function (s, i) { if (s && s.offsetTop <= y) current = i; });
    links.forEach(function (a, i) { a.classList.toggle("is-on", i === current); });
  }
  document.addEventListener("scroll", mark, { passive: true });
  mark();
})();
