"""
Including some customized plot
"""

from .basic_import import *
import matplotlib
import matplotlib.pyplot as plt


def bar_plot(sizes, labels, bar_ylabel, bar_title, colors=None, with_show=False):
    """
    Draw barplot with count on top
    Code from https://www.kaggle.com/phunghieu/a-quick-simple-eda
    """
    y_pos = np.arange(len(labels))
    barlist = plt.bar(y_pos, sizes, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel(bar_ylabel)
    plt.title(bar_title)
    if colors is not None:
        for idx, item in enumerate(barlist):
            item.set_color(colors[idx])

    def autolabel(rects):
        """
        Attach a text label above each bar, displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            plt.text(
                rect.get_x() + rect.get_width() / 2., height, \
                '%d' % int(height),
                ha='center', va='bottom', fontweight='bold'
            )

    autolabel(barlist)

    if with_show:
        plt.show()


def bar_plot_from_one_col(df, bar_ylabel='Freq', bar_title='Counting freq', colors=None, with_show=False):
    """
    Barplot given a list of values
    """
    if isinstance(df, pd.DataFrame):
        col = list(df.columns)
        if col.__len__() != 1:
            raise Exception("Input dataframe should have one column only")
        plot_target = deepcopy(df[col]).values
    else: # pd.Series, np.array, list..
        plot_target = deepcopy(df)

    bar_plot_params = {'bar_ylabel': bar_ylabel,
                       'bar_title': bar_title,
                       'colors': colors,
                       'with_show': with_show}
    unique_val, counts = np.unique(plot_target, return_counts=True)
    bar_plot(counts, labels=[label for label in unique_val], **bar_plot_params)




