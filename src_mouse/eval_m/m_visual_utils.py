import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
from eval.visual_utils import FigureStyle
from eval.visual_utils import pval2stars
from collections import Counter
          
from scipy.stats import shapiro, pearsonr, spearmanr
from scipy.stats import gaussian_kde
from eval.visual_utils import FigureStyle
def print_freq(iput_list):
    # Calculate frequencies
    frequency_counts = Counter(iput_list)

    # Print frequencies
    for gene_type, frequency in frequency_counts.items():
        print(f"{gene_type}: {frequency}")

def density_colored_scatter(x_values, y_values, power_transform=0.5, dot_size=2, cmap='turbo', 
                            figsize=(10, 8), colorbar_label='Density', title='', 
                            xlabel='X Values', ylabel='Y Values', output_file=None, plot_y_equals_x=True):
    """
    Generates a density-colored scatter plot for given x and y values.
    ... [rest of the docstring] ...
    """
    print(f"number of samples: {len(x_values)}")
    assert len(x_values) == len(y_values)
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    nan_indices = np.isnan(x_values) | np.isnan(y_values)

    # Filtering out the elements where either array has NaN
    x_values = x_values[~nan_indices]
    y_values = y_values[~nan_indices]
    # Calculate the point density
    xy = np.vstack([x_values, y_values])
    z = gaussian_kde(xy)(xy)
    z = np.power(z, power_transform)  # Applying transformation to density values

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x_values, y_values, z = x_values[idx], y_values[idx], z[idx]

    
    # Apply FigureStyle
    fig_style = FigureStyle()
    fig_style.apply()

    plt.figure(figsize=figsize)
    scatter = plt.scatter(x_values, y_values, c=z, s=dot_size, cmap=cmap)
    plt.colorbar(scatter, label=colorbar_label)

    ax = plt.gca()

    # Perform Shapiro-Wilk test for normality
    shapiro_test_x = shapiro(x_values)
    shapiro_test_y = shapiro(y_values)
    
    # Use Pearson if both x and y are normally distributed, otherwise use Spearman
    if shapiro_test_x.pvalue > 0.05 and shapiro_test_y.pvalue > 0.05:
        coef, p = pearsonr(x_values, y_values)
        correlation_label = 'r'
    else:
        coef, p = spearmanr(x_values, y_values)
        correlation_label = 'rho'
    
    # Format the p-value
    if p < 1e-20:
        p_text = 'p < 1e-20'
    else:
        p_text = f'p = {p:.3e}'
    
    # Add correlation coefficient, type, and formatted p-value to the plot
    ax.text(0.05, 0.95, f'{correlation_label} = {coef:.2f}\n{p_text}', 
            transform=ax.transAxes, fontsize=fig_style.fontsize,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    fig_style.set_titles(ax, title)
    fig_style.set_labels(ax, xlabel, ylabel)
    if plot_y_equals_x:
        max_range = max(max(x_values), max(y_values))
        plt.plot([0, max_range], [0, max_range], 'k--', color="#505050")  # Black dashed line

    # Save to file if output_file is provided
    if output_file:
        fig_style.save_figure(output_file)

    plt.show()
def plot_PCA(gene_emb_mat, 
             gene_type_list_for_color, 
             title='', 
             figsize=(8, 8), 
             output_file=None, 
             legend_loc='upper right',
             legend_dot_size=5,
             show_plot=False
            ):
    # Create an instance of FigureStyle and apply the style
    my_style = FigureStyle(markersize=0.3)
    my_style.apply()
    print_freq(gene_type_list_for_color)
    # Standardizing the features
    gene_emb_npy_std = StandardScaler().fit_transform(gene_emb_mat)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(gene_emb_npy_std)
    explained_variance = pca.explained_variance_ratio_ * 100  # Variance explained by each component in percentage

    # Creating a DataFrame for the plot
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['Gene Type'] = gene_type_list_for_color

    # Plotting
    plt.figure(figsize=figsize)
    ax = plt.gca()  # Get the current Axes instance
    gene_type_list_for_color_uniq = np.unique(gene_type_list_for_color)
    if all(elem in ['Anchor', 'Human', 'Mouse'] for elem in gene_type_list_for_color_uniq) and len(gene_type_list_for_color_uniq) == 3:
        gene_type_list_for_color_uniq = ['Human','Mouse', 'Anchor']
    for gene_type in gene_type_list_for_color_uniq:
        indicesToKeep = pca_df['Gene Type'] == gene_type
        plt.scatter(pca_df.loc[indicesToKeep, 'PC1'], pca_df.loc[indicesToKeep, 'PC2'], label=gene_type, s=0.03, marker=".")
    my_style.set_labels(ax, f'PC1 ({explained_variance[0]:.2f}%)', f'PC2 ({explained_variance[1]:.2f}%)')
    my_style.set_titles(ax, title)
    legend = my_style.update_legend_style(ax, loc=legend_loc)
    for handle in legend.legendHandles:
        handle.set_sizes([legend_dot_size])
    # Saving the figure if an output path is provided
    if output_file:
        my_style.save_figure(output_file)
    if show_plot:
        plt.show()



def plot_histogram(data, 
                   figsize=(10, 6), 
                   output_file=None,
                   bins=40, 
                   title='', 
                   xlabel='Value', 
                   ylabel='Frequency', 
                   color_index=1, 
                   log_scale=False,
                   y_log_scale=False
                   ):
    """
    Plot a histogram for a given 1D or 2D numpy array with configurable parameters.

    Parameters:
    data (numpy array): A 1D or 2D numpy array for which the histogram is to be plotted.
    figsize (tuple): Size of the figure (width, height).
    bins (int): Number of bins in the histogram.
    title (str): Title of the histogram.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    color_index (int): Index of the color in the palette to use for the histogram.
    log_scale (bool): Whether to use logarithmic scale for the x-axis and binning.
    """
    # Instantiate FigureStyle
    my_style = FigureStyle()
    my_style.apply()

    # Flatten the array if it's 2D
    if len(data.shape) == 2:
        data = data.flatten()

    # Log-transform the data if log_scale is True
    if log_scale:
        # Filtering out non-positive values as they can't be log-transformed
        data = data[data > 0]
        data = np.log10(data)

    # Plotting the histogram
    plt.figure(figsize=figsize)
    ax = plt.gca()  # Get the current Axes instance
    color=my_style.get_palette_colors(3)[color_index]
    ax.hist(data, bins=bins, color=color, edgecolor='black')

    # Set the x-axis to logarithmic scale if specified
    if log_scale:
        # ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(10**x)}'))
    if y_log_scale:
        ax.set_yscale('log')
    # Set style elements
    my_style.set_labels(ax, xlabel, ylabel)
    my_style.set_titles(ax, title)
    if output_file:
        my_style.save_figure(output_file)

    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import ks_2samp


def plot_side_by_side_histogram(df, 
                               figsize=(10, 6), 
                               output_file=None,
                               bins=30, 
                               title='',
                               legend_title='',
                               xlabel='Value', 
                               ylabel='Frequency', 
                               log_scale=False,
                               y_log_scale=False,
                                p_val_pos=(0.25, 0.95)
                               ):
    """
    Plot histograms for multiple groups in a melted DataFrame side by side on the same axis for comparison.

    Parameters:
    df (DataFrame): A DataFrame where each column represents a different group.
    figsize (tuple): Size of the figure (width, height).
    bins (int): Number of bins in the histogram.
    title (str): Title of the histogram.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    log_scale (bool): Whether to use logarithmic scale for the x-axis and binning.
    """
    for column in df.columns:
        # Number of rows with value == 1
        num_rows_value_1 = (df[column] == 1).sum()
        
        # Number of rows with value <= 10
        num_rows_value_le_10 = (df[column] <= 10).sum()
        
        # Total number of rows
        total_rows = df.shape[0]
        
        # Print the information for the current column
        print(f'Column: {column}')
        print(f'Number of rows with value == 1: {num_rows_value_1}')
        print(f'Number of rows with value <= 10: {num_rows_value_le_10}')
        print(f'Total number of rows: {total_rows}')

    columns = df.columns
    if len(df.columns) >= 2:
        for i in range(len(columns)):
            
            for j in range(i + 1, len(columns)):
                stat, p_value = ks_2samp(df[columns[i]], df[columns[j]])
                print(f'KS({columns[i]}, {columns[j]}): stat={stat:.2f}, p={p_value:.2e}')
                ks_text = f'p = {p_value:.2e}'

    # Instantiate FigureStyle
    my_style = FigureStyle()
    my_style.apply()

    # Melt the DataFrame using column names as values in the 'Group' column
    melted_df = df.melt(var_name='Group', value_name='Value')

    # Log-transform the data if log_scale is True
    if log_scale:
        melted_df = melted_df[melted_df['Value'] > 0]  # Filtering out non-positive values
        melted_df['Value'] = np.log10(melted_df['Value'])  # Apply log10 transformation

    # Create the plot
    plt.figure(figsize=figsize)
    ax = sns.histplot(data=melted_df, x='Value', hue='Group', bins=bins, multiple='dodge', shrink=0.8)

    # Set the titles, labels, and other style elements
    my_style.set_labels(ax, xlabel, ylabel)
    my_style.set_titles(ax, title)
    
    if len(df.columns) == 2:
        ax.text(p_val_pos[0], p_val_pos[1], ks_text, transform=ax.transAxes, fontsize=my_style.fontsize,
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    # Set the x-axis to logarithmic scale if specified
    if log_scale:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(10**x)}'))
    if y_log_scale:
        ax.set_yscale('log')
    legend = ax.get_legend()
    if legend_title == None:
        legend_title = legend.get_title().get_text()
    legend.set_title(legend_title, prop={'size': my_style.fontsize, 'weight': 'bold'})                                       
    # Adjusting legend
    # Save to file if output_file is provided
    if output_file:
        my_style.save_figure(output_file)

    plt.show()

def plot_lines(df, x_var, y_vars, y_labels, figsize=(10, 6), ylim_min=None, ylim_max=None, title='', xlabel='Fraction', ylabel='Top K Fraction', output_file=None):
    """
    Plot a line plot with custom style.

    Parameters:
    df (DataFrame): DataFrame containing the data to plot.
    x_var (str): Column name to use for the x-axis.
    y_vars (list): List of column names to plot on the y-axis.
    y_labels (list): List of labels for the y-axis plots.
    figsize (tuple): Size of the figure (width, height), default is (10, 6).
    title (str): Title of the plot, default is 'Top 1 vs Top 10 Fraction by Fraction'.
    xlabel (str): Label for the x-axis, default is 'Fraction'.
    ylabel (str): Label for the y-axis, default is 'Top K Fraction'.
    output_file (str, optional): If provided, the plot will be saved to this file path.
    """
    # Apply figure style
    fig_style = FigureStyle()
    fig_style.apply()

    # Creating the plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    for y_var, label in zip(y_vars, y_labels):
        plt.plot(df[x_var], df[y_var], label=label, marker='o')

    # Apply FigureStyle settings
    fig_style.set_labels(ax, xlabel, ylabel)
    fig_style.set_titles(ax, title)
    fig_style.update_legend_style(ax)
    if ylim_min is not None:
        plt.ylim(bottom=ylim_min)
    if ylim_max is not None:
        plt.ylim(top=ylim_max)
    plt.grid(True)

    # Save to file if output_file is provided
    if output_file:
        fig_style.save_figure(output_file)
    else:
        plt.show()


import textwrap
import matplotlib.gridspec as gridspec

import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
def add_significance_annotation(plot, 
                                data, 
                                x_var, 
                                y_var, 
                                label_column, 
                                group_pairs, 
                                s_config=None, 
                                constant_y=True,
                                horizontal_line_y_pos=0.98,
                                vertical_line_len=0.2,
                                star_y_pos=0.985,
                                ns_y_pos=1.02,
                                non_constant_y_offset_ratio=0.06,
                                pvals = None,
                                use_wilcox_rank_sum_test=False
                               ):
    # Get y_max for all comparisons to determine where to draw the significance lines
    data[x_var] = data[x_var].astype(str)
    y_max = data[y_var].max()
    y_offset = y_max * non_constant_y_offset_ratio # offset for annotation lines
    x_tick_labels = [tick.get_text() for tick in plot.get_xticklabels()]
    print(f"unique group name: {set(data[x_var])}")
    p_val_idx = -1
    for i, (group1_name, group2_name) in enumerate(group_pairs, 1):
        group1_data = data[data[x_var] == group1_name]
        group2_data = data[data[x_var] == group2_name]
        p_val_idx += 1
        if constant_y:
            i = 1

        # Convert to numeric and drop NaNs
        group1 = pd.to_numeric(group1_data[y_var], errors='coerce').dropna()
        group2 = pd.to_numeric(group2_data[y_var], errors='coerce').dropna()
        print(f"group1 shape {group1.shape}")
        print(f"group2 shape {group2.shape}")

        # Perform the independent t-test only if both groups contain data
        if not group1.empty and not group2.empty:
            if pvals == None:
                if use_wilcox_rank_sum_test:
                    statistic, p_value = mannwhitneyu(group1, group2)
                    # Print the results
                    print(f'Statistic: {statistic}, P-value: {p_value}')

                else:
                    _, p_value = stats.ttest_ind(group1, group2)
            else:
                p_value = pvals[p_val_idx]
            print(f"p value: {p_value}")
            sig_star = pval2stars(p_value)  # Ensure pval2stars function is defined or imported

            x1, x2 = x_tick_labels.index(group1_name), x_tick_labels.index(group2_name)
            y, h, col = y_max * horizontal_line_y_pos + i * y_offset, y_offset * vertical_line_len , 'k'  # y location, height, and color for the line
            plot.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=s_config.linewidth, c=col)
            ypos = (y+h)*star_y_pos
            fontsize = 6
            if sig_star == "ns":
                ypos = ypos * ns_y_pos 
                fontsize = 5
            plot.text((x1+x2)*.5, ypos, sig_star, ha='center', va='bottom', color=col, fontsize=fontsize)
            


def plot_comparison_box(data, x_var, y_var='average_diff', 
                        label_column='gene_set', title='',
                        title_font_size=None,
                        group_pairs=None, min_y_value_threshold=0.1, 
                        fig_width=2.3, fig_height=2.8,
                        output_file=None, constant_y=True, 
                        xlab=None, GridSpec=None, ylab=None,
                        x_order=None, log_scale_y=False, 
                        rotate_xticklabels=False,
                        xticklabels_max_len_for_wrapping=10,
                        ylab_max_len_for_wrapping=100,
                        horizontal_line_y_pos=0.98,
                        vertical_line_len=0.2,
                        star_y_pos=0.985,
                        ns_y_pos=1.02,
                        automaticaly_add_one_minus_one_for_y_log_scale=False,
                        desired_y_tick_labels=[0, 5, 10, 20],
                        non_constant_y_offset_ratio=0.08,
                        use_wilcox_rank_sum_test=False
                       ):
    """
    Creates a box plot comparing groups based on a specified column.
    """

    data_sorted = data.sort_values(x_var)
    filtered_df = data_sorted

    sample_counts = data_sorted[x_var].value_counts().sort_index()
    for group_name, count in sample_counts.items():
        print(f'Group {group_name}: {count} samples')

    style_config = FigureStyle()
    style_config.apply()

    # Create the figure with a GridSpec layout
    fig = plt.figure(figsize=(fig_width, fig_height))
    if GridSpec == None:
        gs = gridspec.GridSpec(1, 1, left=0.25, right=0.94, top=0.9, bottom=0.15)
    else:
        print(f"use specified GridSpec {GridSpec}")
        gs = gridspec.GridSpec(1, 1, left=GridSpec[0], right=GridSpec[1], top=GridSpec[2], bottom=GridSpec[3])
    ax = fig.add_subplot(gs[0, 0])
    if automaticaly_add_one_minus_one_for_y_log_scale:
        filtered_df[y_var] = filtered_df[y_var]+1
    # Create the boxplot
    sns.boxplot(ax=ax, x=x_var, y=y_var, data=filtered_df, order=x_order, fliersize=style_config.markersize)

    # ylab = "Clustering Performance Index" if y_var == "average_diff" else y_var
    if log_scale_y:
        ax.set_yscale('log')
        # if automaticaly_add_one_minus_one_for_y_log_scale:
        #     if desired_y_tick_labels == None:
        #         print("Please specify desired_y_tick_labels too!")
        #         return None
        #     transformed_y_ticks = np.log(np.array(desired_y_tick_labels) + 1)
        #     ax.set_yticks(transformed_y_ticks)
        #     # Set your specified y-tick labels
        #     ax.set_yticklabels([str(label) for label in desired_y_tick_labels])
            
    xlab = x_var if xlab is None else xlab
    wrapped_ylab = textwrap.fill(ylab.replace('_', ' '), width=ylab_max_len_for_wrapping)
    style_config.set_labels(ax, xlab, wrapped_ylab)
    style_config.set_titles(ax, title, fontsize=style_config.fontsize+2)

    # Add significance annotation if required
    if group_pairs:
        add_significance_annotation(ax, filtered_df, x_var, y_var, label_column, group_pairs, 
                                    s_config=style_config, 
                                    constant_y=constant_y,
                                    horizontal_line_y_pos=horizontal_line_y_pos,
                                    vertical_line_len=vertical_line_len,
                                    star_y_pos=star_y_pos,
                                    ns_y_pos=ns_y_pos,
                                    non_constant_y_offset_ratio=non_constant_y_offset_ratio,
                                    use_wilcox_rank_sum_test=use_wilcox_rank_sum_test
                                   )
    
    if rotate_xticklabels:
        ax.set_xticklabels([textwrap.fill(label.get_text().replace('_', ' '), width=xticklabels_max_len_for_wrapping) for label in ax.get_xticklabels()], rotation=45, ha='right', va='top', fontsize=style_config.fontsize)
    else:
        ax.set_xticklabels([textwrap.fill(label.get_text().replace('_', ' '), width=xticklabels_max_len_for_wrapping) for label in ax.get_xticklabels()], fontsize=style_config.fontsize)

    style_config.set_titles(ax, title)
    # Save the figure if a file name is provided
    # if output_file is not None:
    #     style_config.save_figure(output_file, tight_layout=False)
    if output_file is not None:
        plt.savefig(output_file, dpi=style_config.dpi)
    plt.show()

def plot_custom_ecdf(data, x_var, hue, col, fig_width, fig_height, 
                     xlim=None, ylim=None, style_config=None, hue_order=None,
                     xlabel=None, ylabel=None,
                     title=None, legend_title=None, legend_title_fontsize=None, legend_label_fontsize=None,
                     output_file=None,
                     ylab_max_len_for_wrapping=38
                    ):
    
    sample_counts = data[x_var].value_counts().sort_index()
    for group_name, count in sample_counts.items():
        print(f'Group {group_name}: {count} samples')

    if style_config is None:
        style_config = FigureStyle()  # Use default style settings
    style_config.apply()

    # Create an ECDF plot
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.ecdfplot(data=data, x=x_var, hue=hue, complementary=True, hue_order=hue_order, legend=True)
    
    # Set title and axis labels (assuming title and labels are provided correctly)
    wrapped_ylab = textwrap.fill(ylabel.replace('_', ' '), width=ylab_max_len_for_wrapping)
    style_config.set_titles(ax, title)
    style_config.set_labels(ax, xlabel, wrapped_ylab)

    # Set xlim and ylim if provided
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if legend_title_fontsize == None:
        legend_title_fontsize = style_config.fontsize
    if legend_label_fontsize == None:
        legend_label_fontsize = style_config.fontsize
    # Retrieve and customize the legend
    legend = ax.get_legend()
    if legend:
        legend.set_title(legend_title if legend_title else "Legend", prop={'size': legend_title_fontsize})  # Set title and font size
        for label in legend.get_texts():
            label.set_fontsize(legend_label_fontsize)  # Set label font size

    # Save the plot to a file if output_file is provided
    if output_file:
        if output_file.endswith('.pdf'):
            style_config.save_figure(output_file, format='pdf')
        else:
            style_config.save_figure(output_file)

    # Show the plot
    plt.show()


            
def plot_comparison_bar_with_ci(df_input, 
                        x_var, 
                        y_var, 
                        hue, 
                        legend_title=None, 
                        output_file=None, 
                        fig_width=3.5, 
                        fig_height=2.8,
                        xlab="",
                        ylab="",
                        title="", 
                        legend_loc='upper right',
                        ci_column='CI',
                        add_legend=True,
                        rotate_xticklabels=True,
                        xticklabels_max_len_for_wrapping=15,
                        group_pairs=None,
                        constant_y=False,  
                        horizontal_line_y_pos=1.05,
                        vertical_line_len=0.2,
                        star_y_pos=0.99,
                        ns_y_pos=1.02,
                        pvals=None,
                        non_constant_y_offset_ratio=0.04,
                        use_wilcox_rank_sum_test=True
                       ):
    # Initialize your style
    style = FigureStyle()
    style.apply()

    df_filt = df_input.copy()

    sample_counts = df_filt[x_var].value_counts().sort_index()
    for group_name, count in sample_counts.items():
        print(f'Group {group_name}: {count} samples')

    # df_filt[hue] = df_filt[hue].astype(str).str.replace('_', ' ')
    # df_filt = df_filt.sort_values([hue, x_var])

    plt.figure(figsize=(fig_width, fig_height))
    
    # Create the bar plot with CI as error bars
    plot = sns.barplot(
        data=df_filt, 
        x=x_var, 
        y=y_var, 
        hue=hue,
        yerr=df_filt[ci_column] if ci_column in df_filt.columns else None  # Add CI as error bars
    )

    style.set_labels(plot, xlab, ylab)
    style.set_titles(plot, title)
    if legend_title is None:
        legend_title = hue
    if add_legend:
        style.update_legend_style(plot, title=legend_title, loc=legend_loc, ncol=1)
    plot.set_yticklabels([label.get_text().replace('_', ' ') for label in plot.get_yticklabels()])
    label_column=None
    if group_pairs:
        add_significance_annotation(plot, df_filt, x_var, y_var, label_column, group_pairs, 
                                    s_config=style, 
                                    constant_y=constant_y,
                                    horizontal_line_y_pos=horizontal_line_y_pos,
                                    vertical_line_len=vertical_line_len,
                                    star_y_pos=star_y_pos,
                                    ns_y_pos=ns_y_pos,
                                    non_constant_y_offset_ratio=non_constant_y_offset_ratio,
                                    pvals=pvals,
                                    use_wilcox_rank_sum_test=use_wilcox_rank_sum_test
                                   )
    if rotate_xticklabels:
        plot.set_xticklabels([textwrap.fill(label.get_text().replace('_', ' '), width=xticklabels_max_len_for_wrapping) for label in plot.get_xticklabels()], rotation=45, ha='right', va='top', fontsize=style.fontsize)
    else:
        plot.set_xticklabels([textwrap.fill(label.get_text().replace('_', ' '), width=xticklabels_max_len_for_wrapping) for label in plot.get_xticklabels()], fontsize=style.fontsize)
    if output_file is not None:
        style.save_figure(output_file)
    
    # Show the plot
    plt.show()