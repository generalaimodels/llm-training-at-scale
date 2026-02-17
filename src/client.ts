import mermaid from "mermaid";
import renderMathInElement from "katex/contrib/auto-render";

type ThemeMode = "ivory" | "graphite";

const THEME_STORAGE_KEY = "docs-theme-mode";

function getThemeButton(): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>("#theme-toggle");
}

function applyTheme(mode: ThemeMode): void {
  document.body.dataset.theme = mode;
  const button = getThemeButton();

  if (button) {
    button.textContent = mode === "ivory" ? "Graphite" : "Ivory";
  }
}

function initThemeToggle(): void {
  const saved = window.localStorage.getItem(THEME_STORAGE_KEY);

  if (saved === "graphite" || saved === "ivory") {
    applyTheme(saved);
  } else {
    applyTheme("ivory");
  }

  const button = getThemeButton();

  if (!button) {
    return;
  }

  button.addEventListener("click", () => {
    const nextMode: ThemeMode = document.body.dataset.theme === "graphite" ? "ivory" : "graphite";
    applyTheme(nextMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, nextMode);
  });
}

function initMenuToggle(): void {
  const menuButton = document.querySelector<HTMLButtonElement>("#menu-toggle");

  if (!menuButton) {
    return;
  }

  menuButton.addEventListener("click", () => {
    document.body.classList.toggle("sidebar-open");
  });
}

function initDocumentFilter(): void {
  const filter = document.querySelector<HTMLInputElement>("#nav-filter");
  const links = Array.from(document.querySelectorAll<HTMLElement>("[data-doc-link]"));

  if (!filter || links.length === 0) {
    return;
  }

  filter.addEventListener("input", () => {
    const query = filter.value.trim().toLowerCase();

    for (const link of links) {
      const text = (link.textContent ?? "").toLowerCase();
      const visible = query.length === 0 || text.includes(query);
      link.classList.toggle("is-filter-hidden", !visible);
    }
  });
}

function initCodeCopyButtons(): void {
  if (!navigator.clipboard?.writeText) {
    return;
  }

  const wrappers = Array.from(document.querySelectorAll<HTMLElement>(".code-shell"));

  for (const wrapper of wrappers) {
    const code = wrapper.querySelector<HTMLElement>("pre code");

    if (!code) {
      continue;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-btn";
    button.textContent = "Copy";

    button.addEventListener("click", async () => {
      const raw = code.textContent ?? "";

      if (!raw) {
        return;
      }

      try {
        await navigator.clipboard.writeText(raw);
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      } catch {
        button.textContent = "Error";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      }
    });

    wrapper.append(button);
  }
}

function initTableOfContentsHighlight(): void {
  const headings = Array.from(
    document.querySelectorAll<HTMLElement>(".doc-content h1[id], .doc-content h2[id], .doc-content h3[id], .doc-content h4[id]")
  );
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>("[data-toc-link]"));

  if (headings.length === 0 || links.length === 0) {
    return;
  }

  const linkByHash = new Map<string, HTMLAnchorElement>();

  for (const link of links) {
    const href = link.getAttribute("href");

    if (href?.startsWith("#")) {
      linkByHash.set(href.slice(1), link);
    }
  }

  const setActive = (id: string): void => {
    for (const link of links) {
      link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`);
    }
  };

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((left, right) => left.boundingClientRect.top - right.boundingClientRect.top);

      if (visible.length > 0) {
        const heading = visible[0].target as HTMLElement;
        setActive(heading.id);
      }
    },
    {
      rootMargin: "-30% 0px -55% 0px",
      threshold: [0, 0.1, 0.8]
    }
  );

  for (const heading of headings) {
    observer.observe(heading);
    const match = linkByHash.get(heading.id);

    if (match) {
      match.addEventListener("click", () => {
        document.body.classList.remove("sidebar-open");
      });
    }
  }
}

function initRevealAnimation(): void {
  const candidates = Array.from(document.querySelectorAll<HTMLElement>(".doc-content > *"));

  if (candidates.length === 0) {
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) {
          continue;
        }

        (entry.target as HTMLElement).classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    },
    {
      rootMargin: "0px 0px -8% 0px",
      threshold: 0.08
    }
  );

  for (const candidate of candidates) {
    candidate.classList.add("reveal");
    observer.observe(candidate);
  }
}

function initMathFallbackRendering(): void {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
      { left: "\\begin{equation}", right: "\\end{equation}", display: true },
      { left: "\\begin{equation*}", right: "\\end{equation*}", display: true },
      { left: "\\begin{align}", right: "\\end{align}", display: true },
      { left: "\\begin{align*}", right: "\\end{align*}", display: true },
      { left: "\\begin{gather}", right: "\\end{gather}", display: true },
      { left: "\\begin{gather*}", right: "\\end{gather*}", display: true },
      { left: "\\begin{multline}", right: "\\end{multline}", display: true },
      { left: "\\begin{multline*}", right: "\\end{multline*}", display: true }
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoredClasses: ["katex", "katex-display"],
    throwOnError: false,
    strict: "ignore"
  });
}

function getMermaidTheme(): "base" | "dark" {
  return document.body.dataset.theme === "graphite" ? "dark" : "base";
}

async function renderMermaidBlock(block: HTMLElement, sequence: number): Promise<void> {
  const source = block.textContent?.trim();

  if (!source) {
    return;
  }

  const mount = document.createElement("figure");
  mount.className = "mermaid-figure";

  const graph = document.createElement("div");
  graph.className = "mermaid-render";
  mount.append(graph);

  try {
    const id = `mermaid-diagram-${sequence}`;
    const rendered = await mermaid.render(id, source);
    graph.innerHTML = rendered.svg;
    block.replaceWith(mount);
  } catch {
    block.classList.add("mermaid-error");
  }
}

async function initMermaid(): Promise<void> {
  const blocks = Array.from(document.querySelectorAll<HTMLElement>("pre.mermaid"));

  if (blocks.length === 0) {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: getMermaidTheme(),
    fontFamily: "Manrope, sans-serif"
  });

  for (let index = 0; index < blocks.length; index += 1) {
    await renderMermaidBlock(blocks[index], index);
  }
}

function initActiveDocLinkScroll(): void {
  const active = document.querySelector<HTMLElement>(".doc-link.is-active");

  if (!active) {
    return;
  }

  active.scrollIntoView({ block: "center" });
}

async function bootstrap(): Promise<void> {
  initThemeToggle();
  initMenuToggle();
  initDocumentFilter();
  initCodeCopyButtons();
  initTableOfContentsHighlight();
  initRevealAnimation();
  initMathFallbackRendering();
  initActiveDocLinkScroll();
  await initMermaid();
}

void bootstrap();
