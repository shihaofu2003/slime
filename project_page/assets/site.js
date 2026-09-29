const metricCopy = {
  pass_at_1: {
    summary: "pass@1：成功轨迹数 149 → 220 → 246 → 252，分母均为 788。Raw 到最终 OPD 的相对提升为 69.13%。",
    stages: "Raw、SFT、Mix RL、OPD 的总体 pass@1 分别为 18.91%、27.92%、31.22%、31.98%。",
    domains: "四阶段分域 pass@1。最终 OPD：Airline 51.25%、Retail 65.00%、Telecom 54.38%、Banking 5.15%。"
  },
  pass_at_4_any: {
    summary: "pass@4(any)：四次至少成功一次的任务数 63 → 85 → 96 → 95，分母均为 197。OPD 比 Mix RL 少 1 题；Raw 到最终 OPD 的相对提升为 50.79%。",
    stages: "Raw、SFT、Mix RL、OPD 的总体 pass@4(any) 分别为 31.98%、43.15%、48.73%、48.22%。",
    domains: "四阶段分域 pass@4(any)。最终 OPD：Airline 75.00%、Retail 85.00%、Telecom 87.50%、Banking 11.34%。"
  },
  pass_power_4: {
    summary: "pass^4：四次全部成功的任务数 17 → 26 → 27 → 28，分母均为 197。Raw 到最终 OPD 的相对提升为 64.71%。",
    stages: "Raw、SFT、Mix RL、OPD 的总体 pass^4 分别为 8.63%、13.20%、13.71%、14.21%。",
    domains: "四阶段分域 pass^4。最终 OPD：Airline 20.00%、Retail 47.50%、Telecom 12.50%、Banking 0.00%。"
  }
};

document.querySelectorAll('input[name="metric"]').forEach(input => {
  input.addEventListener("change", () => {
    const metric = input.value;
    for (const name of ["stages", "domains"]) {
      const path = `assets/figures/${name}-${metric}`;
      const chart = document.getElementById(`${name}-chart`);
      chart.src = `${path}.svg`;
      chart.alt = metricCopy[metric][name];
      document.getElementById(`${name}-large`).href = `${path}.svg`;
      document.getElementById(`${name}-download`).href = `${path}.png`;
    }
    document.getElementById("metric-summary").textContent = metricCopy[metric].summary;
  });
});

const chapterLinks = [...document.querySelectorAll('.sidebar nav a')];
const chapterObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    for (const link of chapterLinks) {
      const active = link.hash === `#${entry.target.id}`;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  }
}, {rootMargin: "-12% 0px -65% 0px", threshold: 0});
chapterLinks.forEach(link => chapterObserver.observe(document.querySelector(link.hash)));
