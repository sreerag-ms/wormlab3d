def interactive_plots():
    """Puts matplotlib into interactive mode by switching to the Qt5 backend."""
    import matplotlib
    gui_backend = 'Qt5Agg'
    matplotlib.use(gui_backend, force=True)


def clear_axes(ax):
    """Removes the ticks from a 3D plot"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])
