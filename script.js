const tabButtons = document.querySelectorAll(".tab-button");
const panels = document.querySelectorAll(".tab-panel");
const dayToggles = document.querySelectorAll(".day-toggle");
const flowNodes = document.querySelectorAll(".flow-node[data-target]");
const glossarySearch = document.querySelector("#glossarySearch");
const detailSearch = document.querySelector("#detailSearch");
const practiceInputs = document.querySelectorAll("[data-practice]");

function activateTab(tabName) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
});

dayToggles.forEach((toggle) => {
  toggle.addEventListener("click", () => {
    const card = toggle.closest(".day-card");
    const expanded = card.classList.toggle("expanded");
    toggle.setAttribute("aria-expanded", String(expanded));
  });
});

document.querySelectorAll(".side-nav a").forEach((link) => {
  link.addEventListener("click", () => activateTab("path"));
});

flowNodes.forEach((node) => {
  node.addEventListener("click", () => {
    const target = document.getElementById(node.dataset.target);
    if (!target) return;
    if (node.dataset.target.startsWith("module-")) activateTab("source");
    if (node.dataset.target.startsWith("day")) activateTab("path");
    setTimeout(() => target.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  });
});

document.querySelectorAll(".code-block").forEach((block) => {
  const button = document.createElement("button");
  button.className = "copy-code";
  button.type = "button";
  button.textContent = "复制";
  button.addEventListener("click", async () => {
    const text = block.innerText.replace(/^复制\s*/, "");
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "已复制";
      setTimeout(() => {
        button.textContent = "复制";
      }, 1200);
    } catch (error) {
      button.textContent = "复制失败";
      setTimeout(() => {
        button.textContent = "复制";
      }, 1200);
    }
  });
  block.prepend(button);
});

if (glossarySearch) {
  glossarySearch.addEventListener("input", () => {
    const keyword = glossarySearch.value.trim().toLowerCase();
    document.querySelectorAll(".glossary-card").forEach((card) => {
      const haystack = `${card.innerText} ${card.dataset.keywords || ""}`.toLowerCase();
      card.style.display = haystack.includes(keyword) ? "" : "none";
    });
  });
}

if (detailSearch) {
  detailSearch.addEventListener("input", () => {
    const keyword = detailSearch.value.trim().toLowerCase();
    document.querySelectorAll("[data-detail-item]").forEach((item) => {
      const haystack = (item.dataset.detailText || item.innerText).toLowerCase();
      item.hidden = keyword.length > 0 && !haystack.includes(keyword);
    });
    document.querySelectorAll("[data-detail-link]").forEach((link) => {
      const haystack = link.innerText.toLowerCase();
      link.hidden = keyword.length > 0 && !haystack.includes(keyword);
    });
  });
}

practiceInputs.forEach((input) => {
  const key = `edurag-practice-${input.dataset.practice}`;
  input.checked = localStorage.getItem(key) === "true";
  input.addEventListener("change", () => {
    localStorage.setItem(key, String(input.checked));
  });
});
