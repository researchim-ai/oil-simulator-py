window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['noscript', 'style', 'textarea', 'pre', 'code'],
    renderActions: {
      addMenu: []
    }
  }
};

document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});

window.addEventListener('beforeprint', () => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});

