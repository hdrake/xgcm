window.MathJax = {
  tex: {
    // Hand-written .md pages go through pymdownx.arithmatex (generic), which
    // emits \(...\) / \[...\]. Notebook pages rendered by mkdocs-jupyter leave
    // math as raw $...$ / $$...$$, so accept both delimiter styles.
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  }
  // No ignoreHtmlClass/processHtmlClass restriction: the arithmatex spans and
  // the notebook cells both need typesetting, and MathJax skips code/pre by
  // default so literal $ in code is left alone.
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})
