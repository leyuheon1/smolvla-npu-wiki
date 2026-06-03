(function () {
    let initializedMermaid = false;

    function prepareMermaid() {
        if (!window.mermaid || initializedMermaid) {
            return;
        }

        window.mermaid.initialize({
            startOnLoad: false,
            securityLevel: "loose"
        });
        initializedMermaid = true;
    }

    function styleContainer(container) {
        container.style.width = "100%";
        container.style.height = "75vh";
        container.style.minHeight = "480px";
        container.style.border = "1px solid var(--md-default-fg-color--lightest)";
        container.style.borderRadius = "6px";
        container.style.overflow = "hidden";
        container.style.background = "var(--md-default-bg-color)";
    }

    async function renderMermaidBlock(block, index) {
        if (block.dataset.rendered === "true") {
            return;
        }

        const code = block.querySelector("code");
        const source = code ? code.textContent : block.textContent;
        const wrapper = document.createElement("div");
        const diagram = document.createElement("div");
        const renderId = "mermaid-zoom-" + index + "-" + Date.now();

        block.dataset.rendered = "true";
        wrapper.className = "mermaid-zoom";
        diagram.className = "mermaid-zoom__diagram";
        styleContainer(diagram);
        wrapper.appendChild(diagram);
        block.replaceWith(wrapper);

        const rendered = await window.mermaid.render(renderId, source);
        diagram.innerHTML = rendered.svg;

        const svg = diagram.querySelector("svg");
        if (!svg || typeof window.svgPanZoom !== "function") {
            return;
        }

        svg.style.width = "100%";
        svg.style.height = "100%";
        svg.style.cursor = "grab";

        window.svgPanZoom(svg, {
            zoomEnabled: true,
            controlIconsEnabled: true,
            fit: true,
            center: true,
            minZoom: 0.25,
            maxZoom: 20
        });
    }

    async function renderMermaidZoom() {
        if (!window.mermaid || typeof window.svgPanZoom !== "function") {
            window.setTimeout(renderMermaidZoom, 200);
            return;
        }

        prepareMermaid();

        const blocks = Array.from(document.querySelectorAll("pre.mermaid-source"));
        await Promise.all(blocks.map(renderMermaidBlock));
    }

    if (window.document$ && typeof window.document$.subscribe === "function") {
        window.document$.subscribe(renderMermaidZoom);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", renderMermaidZoom);
    } else {
        renderMermaidZoom();
    }
})();
