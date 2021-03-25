def interactive_plots():
    """Puts matplotlib into interactive mode by switching to the Qt5 backend."""
    import matplotlib
    gui_backend = 'Qt5Agg'
    matplotlib.use(gui_backend, force=True)
