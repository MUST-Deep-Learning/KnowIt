from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score

import matplotlib.widgets as widgets
from matplotlib.widgets import RangeSlider, CheckButtons, RadioButtons, Button, TextBox, SpanSelector
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnnotationBbox, TextArea
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def clear_all_insets_from_axes(axes_list):

    for ax in axes_list:

        if hasattr(ax, 'child_axes') and ax.child_axes:
            child_axes_copy = list(ax.child_axes)
            for child_ax in child_axes_copy:
                try:
                    child_ax.remove()
                except Exception:
                    pass
            ax.child_axes.clear()


        if hasattr(ax, 'inset_axes'):
            try:
                if isinstance(ax.inset_axes, list):
                    inset_axes_copy = list(ax.inset_axes)
                    for inset_ax in inset_axes_copy:
                        try:
                            inset_ax.remove()
                        except:
                            pass
                    ax.inset_axes.clear()
                else:
                    if ax.inset_axes is not None:
                        try:
                            ax.inset_axes.remove()
                        except:
                            pass
                        ax.inset_axes = None
            except:
                pass


        if hasattr(ax, '_custom_insets'):
            for inset in ax._custom_insets:
                try:
                    inset.remove()
                except:
                    pass
            ax._custom_insets.clear()


f_path = '/home/leah/MUST 2024_NEW/KnowIt/temp_experiments/models/my_new_model/interpretations/DeepLift-eval-random-100-True-123-(1583, 1683).pickle'

#f_path = '/home/leah/MUST 2024_NEW/KnowIt/temp_experiments/models/new3/interpretations/DeepLift-eval-random-100-True-123-(6648, 6748).pickle'
with open(f_path, 'rb') as file:
    tot35 = pickle.load(file)
print(tot35)
input_feat = tot35['input_features'].detach().numpy()
print(input_feat)
attributions = tot35['results'][(0, 0)]['attributions']
att = attributions.detach().numpy()

pred = np.array(tot35['predictions']).flatten()
targets = np.array(tot35['targets']).flatten()
num_predictions = len(pred)

abs_errors = np.abs(pred - targets)

percentile = 68
abs_error_threshold = np.percentile(abs_errors, percentile)

underperforming_indices = np.where(abs_errors > abs_error_threshold)[0]
good_indices = np.where(abs_errors <= abs_error_threshold)[0]

num_time_steps = att.shape[1]

time_steps = np.arange(-5, 6)

min_time = -5.5
max_time = 5.5
feature_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']

fig = plt.figure(figsize=(18, 13))
fig.canvas.manager.set_window_title('Feature Attribution Visualization')

control_gs = gridspec.GridSpec(1, 1, figure=fig)
control_gs.update(top=0.97, bottom=0.82, left=0.05, right=0.95)

feature_gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 0.7])
feature_gs.update(top=0.70, bottom=0.30, left=0.05, right=0.95)  # Further reduced top value

prediction_gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])

prediction_gs.update(top=0.70, bottom=0.30, left=0.05, right=0.95)

input_checkbox_gs = gridspec.GridSpec(1, 1, figure=fig)
input_checkbox_gs.update(top=0.70, bottom=0.30, left=0.70, right=0.95)  # Positioned to the right


ax_input_checkbox = fig.add_subplot(input_checkbox_gs[0, 0])
ax_input_checkbox.set_visible(False)
input_checkbox_grid = None
ax_scatter = fig.add_subplot(control_gs[0, 0])
ax_scatter.set_title("Prediction Values", fontsize=14)
ax_scatter.set_xlabel("Index", fontsize=12)
ax_scatter.set_ylabel("Value", fontsize=12)

ax_features = []
for i in range(4):
    row, col = i // 2, i % 2
    ax = fig.add_subplot(feature_gs[row, col])
    ax_features.append(ax)

ax_checkbox = fig.add_subplot(feature_gs[0:2, 2])
ax_checkbox.set_title("Select Predictions", fontsize=12)
ax_checkbox.set_axis_off()

ax_predictions = []
for i in range(12):
    row, col = i // 4, i % 4
    ax = fig.add_subplot(prediction_gs[row, col])
    ax_predictions.append(ax)
    ax.set_visible(False)

input_gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1])
input_gs.update(top=0.90, bottom=0.40, left=0.05, right=0.95)  # Updated position


ax_input_top = fig.add_subplot(input_gs[0, :])
ax_input_top.set_visible(False)


ax_inputs = []
for row in range(1, 4):
    for col in range(4):
        ax = fig.add_subplot(input_gs[row, col])
        ax_inputs.append(ax)
        ax.set_visible(False)


input_view_selected_indices = []
input_view_current_page = 0
input_view_plots_per_page = 6
input_view_span = None
input_view_ax_prev = None
input_view_ax_next = None
input_view_ax_page_info = None
input_view_prev_button = None
input_view_next_button = None
input_view_page_text = None
ax_prev = plt.axes([0.35, 0.10, 0.1, 0.03])
prev_button = Button(ax_prev, 'Previous')
ax_prev.set_visible(False)

ax_next = plt.axes([0.55, 0.10, 0.1, 0.03])
next_button = Button(ax_next, 'Next')
ax_next.set_visible(False)

ax_page_info = plt.axes([0.45, 0.10, 0.1, 0.03])
ax_page_info.set_axis_off()
ax_page_info.set_visible(False)
page_text = ax_page_info.text(0.5, 0.5, "Page 1/1", ha='center', va='center', fontsize=10)

min_pred = float(min(pred))
max_pred = float(max(pred))

is_updating_slider = False


class CleanRangeSlider(RangeSlider):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.valtext.set_visible(False)
        for label in self.ax.get_xticklabels():
            label.set_visible(False)
        self.ax.set_xticks([])


ax_slider = plt.axes([0.2, 0.14, 0.6, 0.02], facecolor='lightgoldenrodyellow')

range_slider = CleanRangeSlider(
    ax_slider,
    "",
    min_pred,
    max_pred,
    valinit=(min_pred, max_pred)
)


ax_slider.set_xticks([])
ax_slider.set_yticks([])
ax_slider.set_xticklabels([])
ax_slider.set_yticklabels([])
ax_slider.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

ax_slider.set_xlabel("")
ax_slider.set_ylabel("")

ax_slider.spines['top'].set_visible(False)
ax_slider.spines['right'].set_visible(False)
ax_slider.spines['bottom'].set_visible(False)
ax_slider.spines['left'].set_visible(False)

ax_range_text = plt.axes([0.2, 0.16, 0.6, 0.03])
ax_range_text.set_axis_off()
range_text = ax_range_text.text(
    0.5, 0.5,
    f"Range: [{min_pred:.3f}, {max_pred:.3f}] - {num_predictions} predictions",
    ha='center', va='center', fontsize=10
)

ax_filter = plt.axes([0.24, 0.10, 0.1, 0.03])
filter_button = widgets.Button(ax_filter, 'Apply Filter')

view_modes = ['Feature View', 'Prediction View', 'Cluster View', 'Input View']
radio_ax = plt.axes([0.7, 0.05, 0.25, 0.06])
view_radio = RadioButtons(radio_ax, view_modes, active=0)

try:
    for i, (text, circle) in enumerate(zip(view_radio.labels, view_radio.circles)):
        text.set_position((circle.center[0] + 0.15, circle.center[1]))
        text.set_fontsize(9)
        circle.set_radius(0.05)
except AttributeError:

    for text in view_radio.labels:
        text.set_fontsize(9)

ax_all = plt.axes([0.25, 0.05, 0.05, 0.025])
all_button = widgets.Button(ax_all, 'All')
all_button.label.set_fontsize(9)

ax_good = plt.axes([0.31, 0.05, 0.05, 0.025])
good_button = widgets.Button(ax_good, 'Good')
good_button.label.set_fontsize(9)

ax_bad = plt.axes([0.37, 0.05, 0.05, 0.025])
bad_button = widgets.Button(ax_bad, 'Bad')
bad_button.label.set_fontsize(9)

ax_cluster = plt.axes([0.5, 0.05, 0.15, 0.03])
cluster_button = widgets.Button(ax_cluster, 'Run Clustering')
cluster_button.label.set_fontsize(10)
ax_cluster.set_visible(False)


ax_cluster_count = plt.axes([0.5, 0.01, 0.15, 0.03])
cluster_count_textbox = TextBox(ax_cluster_count, "Clusters:", initial="5")
cluster_count_textbox.label.set_fontsize(9)
ax_cluster_count.set_visible(False)



def validate_cluster_count(text):
    if text.lower() == 'none':
        return True
    try:
        value = int(text)
        return value > 0
    except ValueError:
        return False



def on_cluster_count_submit(text):
    if not validate_cluster_count(text):
        cluster_count_textbox.set_val("5")
    else:

        cluster_count_textbox.set_val(text if text.lower() != 'none' else "None")



cluster_count_textbox.on_submit(on_cluster_count_submit)


filtered_indices = np.arange(num_predictions)
current_view = 'Feature View'
feature_view_active = True
feature_lines = [[] for _ in range(4)]
prediction_lines = [[] for _ in range(num_predictions)]
current_page = 0
plots_per_page = 12
scatter_pred = None
current_perf_filter = 'all'


cluster_results = None

cluster_axes = None

checkbox_grid = None
checkbox_active = np.zeros(128, dtype=bool)
checkbox_indices = np.zeros(128, dtype=int)
checkbox_labels = []
NUM_ROWS = 32
NUM_COLS = 4
CHECKBOXES_PER_PAGE = NUM_ROWS * NUM_COLS


cluster_checkbox_grid = None

tooltip = ax_scatter.annotate("",
                              xy=(0, 0),
                              xytext=(20, 20),
                              textcoords="offset points",
                              bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                              arrowprops=dict(arrowstyle="->"),
                              visible=False)


def update_cluster_range_slider(cluster_id):

    global range_slider, is_updating_slider


    if is_updating_slider:
        return False

    is_updating_slider = True

    try:

        if 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
            overall_clusters = cluster_results['overall']
            cluster_indices = [idx for idx in range(len(overall_clusters)) if overall_clusters[idx] == cluster_id]

            if cluster_indices:

                cluster_min = min(pred[cluster_indices])
                cluster_max = max(pred[cluster_indices])


                range_buffer = (cluster_max - cluster_min) * 0.05
                cluster_min = max(min_pred, cluster_min - range_buffer)
                cluster_max = min(max_pred, cluster_max + range_buffer)


                range_slider.valmin = cluster_min
                range_slider.valmax = cluster_max
                range_slider.set_val((cluster_min, cluster_max))


                filter_text = "All"
                if current_perf_filter == 'good':
                    filter_text = "Well performing"
                elif current_perf_filter == 'bad':
                    filter_text = "Underperforming"

                range_text.set_text(
                    f"Cluster {cluster_id} Range: [{cluster_min:.3f}, {cluster_max:.3f}] - {len(cluster_indices)} {filter_text} predictions"
                )


                ax_slider.set_xticks([])
                ax_slider.set_yticks([])
                ax_slider.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

                return True
    finally:
        is_updating_slider = False

    return False


def apply_performance_filter():


    min_val, max_val = range_slider.val
    range_indices = np.where((pred >= min_val) & (pred <= max_val))[0]


    if current_view == 'Cluster View' and 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
        overall_clusters = cluster_results['overall']
        unique_clusters = np.array(sorted(np.unique(overall_clusters)))

        if len(unique_clusters) > 0 and current_page < len(unique_clusters):
            current_cluster = unique_clusters[current_page]

            cluster_indices = [idx for idx in range_indices if overall_clusters[idx] == current_cluster]
            range_indices = np.array(cluster_indices)


    if current_perf_filter == 'good':

        filtered = np.intersect1d(range_indices, good_indices)
    elif current_perf_filter == 'bad':

        filtered = np.intersect1d(range_indices, underperforming_indices)
    else:
        filtered = range_indices


    filtered = sorted(filtered, key=lambda x: pred[x])

    return filtered



def update_scatter_plot(filtered_indices=None):

    global scatter_pred

    ax_scatter.clear()

    if filtered_indices is None or len(filtered_indices) == 0:

        indices = np.arange(num_predictions)
        scatter_pred = ax_scatter.scatter(indices, pred, color='blue', alpha=0.7, label='Predictions', picker=5)

        if current_perf_filter == 'all':
            ax_scatter.scatter(indices, targets, color='green', alpha=0.5, label='Targets', marker='x')
    else:

        scatter_pred = ax_scatter.scatter(filtered_indices, pred[filtered_indices], color='blue', alpha=0.7,
                                          label='Predictions', picker=5)

        if current_perf_filter == 'all':
            ax_scatter.scatter(filtered_indices, targets[filtered_indices], color='green', alpha=0.5,
                               label='Targets', marker='x')


        if current_perf_filter in ['all', 'bad']:

            under_indices = np.intersect1d(filtered_indices, underperforming_indices)
            if len(under_indices) > 0:
                ax_scatter.scatter(under_indices, pred[under_indices], color='red', alpha=0.9,
                                   label='Underperforming', marker='o', edgecolors='black', s=80)

    ax_scatter.set_title("Prediction Values", fontsize=14)
    ax_scatter.set_xlabel("Index", fontsize=12, labelpad=10)
    ax_scatter.set_ylabel("Value", fontsize=12)
    ax_scatter.legend(loc='upper right')
    ax_scatter.grid(True, alpha=0.3)


    ax_scatter.axhline(y=0, color='gray', linestyle='--', alpha=0.5)


    tooltip.set_visible(False)


def toggle_view_mode(view_mode):

    global feature_view_active, current_view

    current_view = view_mode


    for ax in ax_features + ax_predictions:
        ax.clear()
        ax.set_visible(False)


    ax_prev.set_visible(False)
    ax_next.set_visible(False)
    ax_page_info.set_visible(False)


    ax_cluster.set_visible(False)


    ax_cluster_count.set_visible(False)

    if view_mode == 'Feature View':
        feature_view_active = True

        for ax in ax_features:
            ax.set_visible(True)
        ax_checkbox.set_visible(True)
        ax_filter.set_visible(True)

        update_feature_view()
    elif view_mode == 'Prediction View':
        feature_view_active = False

        ax_checkbox.set_visible(False)
        ax_filter.set_visible(False)
        ax_prev.set_visible(True)
        ax_next.set_visible(True)
        ax_page_info.set_visible(True)

        update_prediction_view()
    elif view_mode == 'Cluster View':
        feature_view_active = False

        ax_prev.set_visible(True)
        ax_next.set_visible(True)
        ax_page_info.set_visible(True)

        ax_checkbox.set_visible(True)

        ax_filter.set_visible(True)

        ax_cluster.set_visible(True)

        ax_cluster_count.set_visible(True)


        if 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
            overall_clusters = cluster_results['overall']
            unique_clusters = np.array(sorted(np.unique(overall_clusters)))
            if len(unique_clusters) > 0 and current_page < len(unique_clusters):
                current_cluster = unique_clusters[current_page]
                update_cluster_range_slider(current_cluster)


        update_cluster_view(skip_slider_update=False)

    fig.canvas.draw_idle()



class FixedGridCheckButtons:


    def __init__(self, ax, num_rows, num_cols):
        self.ax = ax
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.max_boxes = num_rows * num_cols


        self.row_height = 1.0 / (num_rows + 2)
        self.col_width = 1.0 / num_cols


        self.checkboxes = []
        self.x_marks = []
        self.labels = []
        self.indices = []
        self.values = []
        self.active = {}


        self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

    def update_grid(self, prediction_indices, clear_selection=False):


        self.ax.clear()
        self.ax.set_title("Select Predictions", fontsize=12)
        self.ax.set_axis_off()


        self.checkboxes = []
        self.x_marks = []
        self.labels = []
        self.indices = []
        self.values = []


        if clear_selection:
            self.active = {}


        if not prediction_indices:
            self.ax.text(0.5, 0.5, "No predictions in range",
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=10, color='gray')
            return


        num_to_show = min(len(prediction_indices), self.max_boxes)

        for i in range(num_to_show):

            pred_idx = prediction_indices[i]

            row = i % self.num_rows
            col = i // self.num_rows


            x = col * self.col_width + 0.05
            y = 1.0 - (row + 1) * self.row_height


            is_active = self.active.get(pred_idx, False)


            border_color = 'red' if pred_idx in underperforming_indices else 'black'
            border_width = 1


            checkbox = plt.Rectangle(
                (x, y),
                self.row_height * 0.7,
                self.row_height * 0.7,
                fill=False,
                edgecolor=border_color,
                linewidth=border_width
            )


            x_mark = None
            if is_active:

                checkbox_width = self.row_height * 0.7
                checkbox_height = self.row_height * 0.7
                x_center = x + checkbox_width / 2
                y_center = y + checkbox_height / 2


                x_mark = self.ax.text(
                    x_center,
                    y_center,
                    'X',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold'
                )


            label_text = f"#{pred_idx} ({pred[pred_idx]:.2f})"

            label = self.ax.text(
                x + self.row_height * 0.9,
                y + self.row_height * 0.35,
                label_text,
                fontsize=8,
                va='center'
            )

            # Add to plot and store
            self.ax.add_patch(checkbox)
            self.checkboxes.append(checkbox)
            self.x_marks.append(x_mark)
            self.labels.append(label)
            self.indices.append(pred_idx)
            self.values.append(pred[pred_idx])

        # Add message for more predictions if needed
        if len(prediction_indices) > self.max_boxes:
            remaining = len(prediction_indices) - self.max_boxes
            self.ax.text(
                0.5, 0.02,
                f"{remaining} more predictions not shown",
                ha='center', va='bottom', fontsize=8, color='red',
                transform=self.ax.transAxes
            )

    def on_click(self, event):
        """Handle click events for checkboxes"""

        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        for i, rect in enumerate(self.checkboxes):

            xmin, ymin = rect.get_xy()
            xmax = xmin + rect.get_width()
            ymax = ymin + rect.get_height()

            if xmin <= x <= xmax and ymin <= y <= ymax:

                pred_idx = self.indices[i]

                self.active[pred_idx] = not self.active.get(pred_idx, False)

                checkbox_width = rect.get_width()
                checkbox_height = rect.get_height()
                x_center = xmin + checkbox_width / 2
                y_center = ymin + checkbox_height / 2

                if self.active[pred_idx]:

                    x_mark = self.ax.text(
                        x_center,
                        y_center,
                        'X',
                        ha='center',
                        va='center',
                        fontsize=9,
                        fontweight='bold'
                    )
                    self.x_marks[i] = x_mark
                else:
                    # Remove X mark
                    if self.x_marks[i]:
                        self.x_marks[i].remove()
                        self.x_marks[i] = None

                # UPDATE: Call appropriate update function based on current view
                if current_view == 'Feature View':
                    update_feature_view()
                elif current_view == 'Cluster View':
                    update_cluster_view()
                elif current_view == 'Input View':  # ADD THIS LINE
                    update_input_view()  # ADD THIS LINE

                fig.canvas.draw_idle()
                break

    def get_selected_indices(self):


        return [idx for idx, active in self.active.items() if active]

    def clear_all_selections(self):

        self.active = {}


        for i, x_mark in enumerate(self.x_marks):
            if x_mark:
                x_mark.remove()
                self.x_marks[i] = None

        update_feature_view()
        fig.canvas.draw_idle()



def update_filtered_predictions(val):
    global filtered_indices, checkbox_grid, cluster_checkbox_grid, input_checkbox_grid, current_page, input_view_selected_indices


    current_view_before = current_view


    filtered_indices = apply_performance_filter()


    min_val, max_val = range_slider.val
    filter_text = "All"
    if current_perf_filter == 'good':
        filter_text = "Well performing"
    elif current_perf_filter == 'bad':
        filter_text = "Underperforming"


    if current_view == 'Cluster View' and 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
        overall_clusters = cluster_results['overall']
        unique_clusters = np.array(sorted(np.unique(overall_clusters)))

        if len(unique_clusters) > 0 and current_page < len(unique_clusters):
            current_cluster = unique_clusters[current_page]
            cluster_indices = [idx for idx in filtered_indices if overall_clusters[idx] == current_cluster]
            range_text.set_text(
                f"Cluster {current_cluster} Range: [{min_val:.3f}, {max_val:.3f}] - {len(cluster_indices)} {filter_text} predictions"
            )
        else:
            range_text.set_text(
                f"Range: [{min_val:.3f}, {max_val:.3f}] - {len(filtered_indices)} {filter_text} predictions")
    else:
        range_text.set_text(
            f"Range: [{min_val:.3f}, {max_val:.3f}] - {len(filtered_indices)} {filter_text} predictions")


    update_scatter_plot(filtered_indices)


    if feature_view_active and current_view == 'Feature View':
        if checkbox_grid is None:
            ax_checkbox.clear()
            ax_checkbox.set_title("Select Predictions", fontsize=12)
            ax_checkbox.set_axis_off()
            checkbox_grid = FixedGridCheckButtons(ax_checkbox, NUM_ROWS, NUM_COLS)
        checkbox_grid.update_grid(filtered_indices)


    if current_view == 'Input View':
        if input_checkbox_grid is None:
            ax_input_checkbox.clear()
            ax_input_checkbox.set_title("Select Predictions", fontsize=12)
            ax_input_checkbox.set_axis_off()
            input_checkbox_grid = FixedGridCheckButtons(ax_input_checkbox, NUM_ROWS, NUM_COLS)

        input_checkbox_grid.update_grid(filtered_indices)


    if current_view == 'Cluster View' and 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
        ax_checkbox.set_visible(True)
        overall_clusters = cluster_results['overall']
        unique_clusters = np.array(sorted(np.unique(overall_clusters)))
        if len(unique_clusters) > 0 and current_page < len(unique_clusters):
            current_cluster = unique_clusters[current_page]
            cluster_indices = [idx for idx in filtered_indices if overall_clusters[idx] == current_cluster]
            if cluster_checkbox_grid is None:
                cluster_checkbox_grid = FixedGridCheckButtons(ax_checkbox, NUM_ROWS, NUM_COLS)
            cluster_checkbox_grid.update_grid(cluster_indices)


    if current_view_before == 'Feature View':
        update_feature_view()
    elif current_view_before == 'Prediction View':
        update_prediction_view()
    elif current_view_before == 'Cluster View':
        update_cluster_view(skip_slider_update=True)
    elif current_view_before == 'Input View':

        update_input_view()

    fig.canvas.draw_idle()



def update_feature_view():
    """Update feature view with explicit inset prevention and position reset"""
    if not feature_view_active:
        return


    clear_all_insets_from_axes(ax_features)


    for ax in ax_features:
        if hasattr(ax, '_original_position'):
            ax.set_position(ax._original_position)

    selected_indices = []
    if checkbox_grid is not None:
        selected_indices = checkbox_grid.get_selected_indices()

    for ax in ax_features:
        ax.clear()
        ax.set_visible(True)

    for i, ax in enumerate(ax_features):
        if i < 4:
            ax.set_title(f"{feature_names[i]} Attributions", fontsize=10, pad=15)
            ax.set_xlabel("Time Steps", fontsize=9, labelpad=8)
            ax.set_ylabel("Attribution Value", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            ax.set_xticks(time_steps)
            ax.set_xticklabels(time_steps, fontsize=8)
            ax.set_xlim(min_time, max_time)


            for pred_idx in filtered_indices:
                line_color = feature_colors[i]
                alpha_val = 0.15

                ax.plot(
                    time_steps,
                    att[pred_idx, :, i],
                    color=line_color,
                    alpha=alpha_val,
                    linewidth=1.2,
                    label=None
                )


            if selected_indices:
                for pred_idx in selected_indices:
                    line_color = feature_colors[i]
                    ax.plot(
                        time_steps,
                        att[pred_idx, :, i],
                        color=line_color,
                        alpha=0.9,
                        linewidth=2.5,
                        label=None
                    )

            if not filtered_indices:
                ax.text(0.5, 0.5, "No predictions in range",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')

            if ax.get_legend() is not None:
                ax.get_legend().remove()

    plt.subplots_adjust(hspace=0.6, wspace=0.4)



def update_prediction_view():
    """Update prediction view with explicit inset cleanup and position reset"""
    global current_page

    if feature_view_active:
        return


    clear_all_insets_from_axes(ax_features)
    clear_all_insets_from_axes(ax_predictions)


    for ax in ax_features:
        if hasattr(ax, '_original_position'):
            ax.set_position(ax._original_position)
        ax.clear()
        ax.set_visible(False)


    for ax in ax_predictions:
        ax.set_visible(True)


    if 'cluster_axes' in globals() and cluster_axes is not None:
        for ax in cluster_axes:
            ax.set_visible(False)

    display_preds = filtered_indices
    total_pages = max(1, (len(display_preds) + plots_per_page - 1) // plots_per_page)
    current_page = min(current_page, total_pages - 1)
    current_page = max(0, current_page)

    page_text.set_text(f"Page {current_page + 1}/{total_pages}")


    for ax in ax_predictions:
        ax.clear()

    start_idx = current_page * plots_per_page
    end_idx = min(start_idx + plots_per_page, len(display_preds))
    page_preds = display_preds[start_idx:end_idx]

    prev_button.ax.set_visible(True)
    next_button.ax.set_visible(True)
    ax_page_info.set_visible(True)

    plt.subplots_adjust(wspace=0.5)

    for i, ax in enumerate(ax_predictions):
        if i < len(page_preds):
            ax.set_visible(True)
            pred_idx = page_preds[i]

            title_color = 'red' if pred_idx in underperforming_indices else 'black'

            if pred_idx in underperforming_indices:
                title = f"Pred #{pred_idx} ({pred[pred_idx]:.3f}) [UNDER]"
            else:
                title = f"Pred #{pred_idx} ({pred[pred_idx]:.3f})"

            ax.margins(y=0.05)
            ax.set_title(title, fontsize=9, color=title_color, pad=2)
            ax.set_xlabel("Time", fontsize=8, labelpad=8)
            ax.set_ylabel("Attribution", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            ax.set_xticks(time_steps)
            ax.set_xticklabels(time_steps, fontsize=7)
            ax.set_xlim(min_time, max_time)


            for feat_idx in range(4):
                ax.plot(
                    time_steps,
                    att[pred_idx, :, feat_idx],
                    color=feature_colors[feat_idx],
                    linewidth=1.5,
                    label=None
                )

            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            ax.set_visible(False)

    if not page_preds:
        ax_predictions[0].set_visible(True)
        ax_predictions[0].text(0.5, 0.5, "No predictions to display",
                               ha='center', va='center', transform=ax_predictions[0].transAxes,
                               fontsize=10, color='gray')


def update_cluster_view(skip_slider_update=False):
    global current_page, cluster_checkbox_grid, cluster_gs, cluster_axes

    try:
        print("Updating cluster view...")


        clear_all_insets_from_axes(ax_features)
        clear_all_insets_from_axes(ax_predictions)


        for ax in ax_features:
            ax.clear()
            ax.set_visible(False)


        for ax in ax_predictions:
            ax.clear()
            ax.set_visible(False)


        if cluster_axes is None:
            cluster_gs = gridspec.GridSpec(3, 2, figure=fig)
            cluster_gs.update(top=0.70, bottom=0.30, left=0.05, right=0.65, hspace=0.9, wspace=0.4)

            mean_ax = fig.add_subplot(cluster_gs[0, :])
            feature_axes = [
                fig.add_subplot(cluster_gs[1, 0]),
                fig.add_subplot(cluster_gs[1, 1]),
                fig.add_subplot(cluster_gs[2, 0]),
                fig.add_subplot(cluster_gs[2, 1])
            ]
            cluster_axes = [mean_ax] + feature_axes
        else:
            for ax in cluster_axes:
                ax.clear()
                ax.set_visible(True)

        if 'cluster_results' not in globals() or cluster_results is None:
            print("No cluster results available")
            cluster_axes[0].text(0.5, 0.5, "Please run clustering first",
                                 ha='center', va='center', fontsize=12, color='red',
                                 transform=cluster_axes[0].transAxes)
            return

        if 'overall' not in cluster_results:
            print("No overall clustering in results")
            cluster_axes[0].text(0.5, 0.5, "Clustering results incomplete.\nTry running clustering again.",
                                 ha='center', va='center', fontsize=12, color='red',
                                 transform=cluster_axes[0].transAxes)
            return

        overall_clusters = cluster_results['overall']
        unique_clusters = np.array(sorted(np.unique(overall_clusters)))
        print(f"Found {len(unique_clusters)} unique clusters")

        total_clusters = len(unique_clusters)
        total_pages = total_clusters

        current_page = min(current_page, total_pages - 1)
        current_page = max(0, current_page)

        if total_clusters > 0:
            current_cluster = unique_clusters[current_page]
            if not skip_slider_update:
                update_cluster_range_slider(current_cluster)
        else:
            print("No clusters to show")
            cluster_axes[0].text(0.5, 0.5, "No clusters found",
                                 ha='center', va='center', fontsize=12, color='red',
                                 transform=cluster_axes[0].transAxes)
            return


        if 'davies_bouldin_score' in cluster_results:
            db_score = cluster_results['davies_bouldin_score']
            if not hasattr(fig, 'db_score_text'):
                db_ax = plt.axes([0.7, 0.005, 0.25, 0.02])
                db_ax.set_axis_off()
                fig.db_score_text = db_ax.text(0, 0.5, f"Davies-Bouldin Score: {db_score:.4f}",
                                               fontsize=8, color='gray')
                fig.db_score_ax = db_ax
            else:
                fig.db_score_text.set_text(f"Davies-Bouldin Score: {db_score:.4f}")
                fig.db_score_ax.set_visible(True)
        elif hasattr(fig, 'db_score_ax'):
            fig.db_score_ax.set_visible(False)

        display_preds = filtered_indices
        print(f"Working with {len(display_preds)} filtered predictions")

        cluster_indices = [idx for idx in display_preds if overall_clusters[idx] == current_cluster]
        print(f"Cluster {current_cluster}: {len(cluster_indices)} predictions after filtering")

        page_text.set_text(
            f"Cluster {current_cluster} ({current_page + 1}/{total_pages}) - {len(cluster_indices)} predictions in range"
        )

        if len(cluster_indices) == 0:
            cluster_axes[0].text(0.5, 0.5, f"No predictions in Cluster {current_cluster} after filtering",
                                 ha='center', va='center', fontsize=12, color='gray',
                                 transform=cluster_axes[0].transAxes)

            ax_prev.set_visible(True)
            next_button.ax.set_visible(True)
            ax_next.set_visible(True)
            page_text.set_visible(True)
            ax_page_info.set_visible(True)

            ax_checkbox.set_visible(True)
            if cluster_checkbox_grid is None:
                cluster_checkbox_grid = FixedGridCheckButtons(ax_checkbox, NUM_ROWS, NUM_COLS)
            cluster_checkbox_grid.update_grid([])

            ax_filter.set_visible(True)
            ax_cluster.set_visible(True)
            ax_cluster_count.set_visible(True)
            return

        if isinstance(cluster_indices, np.ndarray):
            cluster_indices = cluster_indices.tolist()

        import matplotlib.cm as cm
        cluster_cmap = cm.get_cmap('tab10', len(unique_clusters))

        has_exemplars = 'exemplars' in cluster_results
        exemplars = cluster_results['exemplars'] if has_exemplars else {}
        exemplar_idx = exemplars.get(current_cluster, None)

        if exemplar_idx is not None:
            print(f"Exemplar for cluster {current_cluster}: #{exemplar_idx}")

        cluster_color_idx = np.where(unique_clusters == current_cluster)[0][0]
        cluster_color = cluster_cmap(cluster_color_idx)

        ax_checkbox.set_visible(True)
        if cluster_checkbox_grid is None:
            cluster_checkbox_grid = FixedGridCheckButtons(ax_checkbox, NUM_ROWS, NUM_COLS)

        cluster_checkbox_grid.update_grid(cluster_indices)
        selected_indices = cluster_checkbox_grid.get_selected_indices()
        print(f"Selected predictions: {selected_indices}")

        mean_ax = cluster_axes[0]
        pos = mean_ax.get_position()
        mean_ax.set_position([
            pos.x0,
            pos.y0 + 0.07,
            pos.width,
            pos.height * 0.9
        ])

        count = len(cluster_indices)
        total_in_cluster = len(
            [idx for idx in range(len(overall_clusters)) if overall_clusters[idx] == current_cluster])

        # Prepare the plot
        mean_ax.set_xlabel("Time Steps", fontsize=10, labelpad=20)
        mean_ax.set_ylabel("Mean Attribution", fontsize=10)
        mean_ax.grid(True, alpha=0.3)
        mean_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        mean_ax.set_title(f"Cluster {current_cluster} - {count}/{total_in_cluster} predictions in range",
                          color=cluster_color, fontsize=12, fontweight='bold', pad=20)

        mean_ax.set_xticks(time_steps)
        mean_ax.set_xticklabels(time_steps, fontsize=8)
        mean_ax.set_xlim(min_time, max_time)
        mean_ax.xaxis.set_label_coords(0.5, -0.25)

        # Plot mean attribution for each feature
        for feat_idx in range(4):
            feature_data = att[cluster_indices, :, feat_idx]
            feature_mean = np.mean(feature_data, axis=0)
            mean_ax.plot(time_steps, feature_mean, color=feature_colors[feat_idx],
                         linewidth=2.5, label=feature_names[feat_idx])

        mean_ax.legend(loc='best', fontsize=9)

        # Plot individual feature plots
        for feat_idx in range(4):
            feat_ax = cluster_axes[feat_idx + 1]
            feat_ax.set_title(f"{feature_names[feat_idx]} - Individual Predictions",
                              fontsize=10, pad=20)
            feat_ax.set_xlabel("Time Steps", fontsize=8, labelpad=15)
            feat_ax.set_ylabel("Attribution", fontsize=8)
            feat_ax.grid(True, alpha=0.3)
            feat_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            feat_ax.xaxis.set_label_coords(0.5, -0.25)
            feat_ax.set_xticks(time_steps)
            feat_ax.set_xticklabels(time_steps, fontsize=7)
            feat_ax.set_xlim(min_time, max_time)


            for pred_idx in cluster_indices:
                if pred_idx == exemplar_idx or pred_idx in selected_indices:
                    continue
                feat_ax.plot(time_steps, att[pred_idx, :, feat_idx],
                             color=feature_colors[feat_idx], alpha=0.15, linewidth=1.0)


            if exemplar_idx is not None and exemplar_idx in cluster_indices:
                feat_ax.plot(time_steps, att[exemplar_idx, :, feat_idx],
                             color=feature_colors[feat_idx], alpha=0.9, linewidth=1.0,
                             label=f"Exemplar #{exemplar_idx}", linestyle='-')


            for pred_idx in selected_indices:
                if pred_idx in cluster_indices:
                    feat_ax.plot(time_steps, att[pred_idx, :, feat_idx],
                                 color=feature_colors[feat_idx], alpha=0.8, linewidth=1.0)


            has_exemplar = exemplar_idx is not None and exemplar_idx in cluster_indices
            if has_exemplar:
                handles, labels = feat_ax.get_legend_handles_labels()
                exemplar_label_idx = [i for i, label in enumerate(labels) if "Exemplar" in label]
                if exemplar_label_idx:
                    feat_ax.legend([handles[exemplar_label_idx[0]]], [labels[exemplar_label_idx[0]]],
                                   loc='best', fontsize=8)

        plt.subplots_adjust(hspace=0.7, wspace=0.4)


        ax_prev.set_visible(True)
        next_button.ax.set_visible(True)
        ax_next.set_visible(True)
        page_text.set_visible(True)
        ax_page_info.set_visible(True)
        ax_filter.set_visible(True)
        ax_cluster.set_visible(True)
        ax_cluster_count.set_visible(True)


        for ax in ax_features + ax_predictions:
            ax.set_visible(False)

        print("Cluster view updated successfully")

    except Exception as e:
        import traceback
        print(f"Error in update_cluster_view: {str(e)}")
        traceback.print_exc()

        if cluster_axes is not None and len(cluster_axes) > 0:
            cluster_axes[0].clear()
            cluster_axes[0].set_visible(True)
            cluster_axes[0].text(0.5, 0.5, f"Error updating cluster view:\n{str(e)}",
                                 ha='center', va='center', fontsize=10, color='red',
                                 transform=cluster_axes[0].transAxes)
        else:
            ax_predictions[0].set_visible(True)
            ax_predictions[0].clear()
            ax_predictions[0].text(0.5, 0.5, f"Error updating cluster view:\n{str(e)}",
                                   ha='center', va='center', fontsize=10, color='red',
                                   transform=ax_predictions[0].transAxes)


def update_input_view():
    """Input View with insets - ONLY HERE and with proper tracking"""
    global input_view_selected_indices, input_view_current_page, input_checkbox_grid




    clear_all_insets_from_axes(ax_features)
    clear_all_insets_from_axes(ax_predictions)
    clear_all_insets_from_axes(ax_inputs)


    ax_input_top.clear()
    ax_input_top.set_visible(False)


    ax_prev.set_visible(False)
    ax_next.set_visible(False)
    ax_page_info.set_visible(False)


    for ax in ax_inputs:
        ax.clear()
        ax.set_visible(False)


    if hasattr(fig, 'ax_span_selector'):
        fig.ax_span_selector.clear()
        fig.ax_span_selector.set_visible(False)


    ax_checkbox.set_visible(False)
    ax_input_checkbox.set_visible(True)


    ax_slider.set_visible(True)
    ax_range_text.set_visible(True)
    ax_filter.set_visible(True)


    if input_checkbox_grid is None:
        ax_input_checkbox.clear()
        ax_input_checkbox.set_title("Select Predictions", fontsize=12)
        ax_input_checkbox.set_axis_off()
        input_checkbox_grid = FixedGridCheckButtons(ax_input_checkbox, NUM_ROWS, NUM_COLS)


    input_checkbox_grid.update_grid(filtered_indices)


    selected_indices = []
    if input_checkbox_grid is not None:
        selected_indices = input_checkbox_grid.get_selected_indices()
        print(f"DEBUG: Selected indices from Input View checkbox: {selected_indices}")

    input_view_selected_indices = selected_indices


    feature_gs.update(top=0.70, bottom=0.30, left=0.05, right=0.95, hspace=0.6, wspace=0.4)


    feature_layout = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]


    for feat_idx in range(4):
        row, col = feature_layout[feat_idx]


        main_ax = ax_features[feat_idx]
        main_ax.clear()
        main_ax.set_visible(True)


        if not hasattr(main_ax, '_custom_insets'):
            main_ax._custom_insets = []


        pos = main_ax.get_position()
        width_expansion_factor = 1.15
        new_width = pos.width * width_expansion_factor


        if col == 1:
            max_right = 0.67
            if pos.x0 + new_width > max_right:
                new_width = max_right - pos.x0


        main_ax.set_position([pos.x0, pos.y0, new_width, pos.height])

        print(f"DEBUG: Expanded feature {feat_idx} width from {pos.width:.3f} to {new_width:.3f}")


        main_ax.set_title(f"{feature_names[feat_idx]} Attributions", fontsize=12, pad=20)
        main_ax.set_xlabel("Time Steps", fontsize=11, labelpad=10)
        main_ax.set_ylabel("Attribution Value", fontsize=11)
        main_ax.grid(True, alpha=0.3)
        main_ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)


        main_ax.set_xticks(time_steps)
        main_ax.set_xticklabels(time_steps, fontsize=9)
        main_ax.set_xlim(min_time, max_time)
        main_ax.tick_params(axis='both', which='major', labelsize=9)


        for pred_idx in filtered_indices:
            if pred_idx in input_view_selected_indices:
                continue

            line_color = feature_colors[feat_idx]
            alpha_val = 0.15 if pred_idx in underperforming_indices else 0.08
            linestyle = '-'

            main_ax.plot(
                time_steps,
                att[pred_idx, :, feat_idx],
                color=line_color,
                alpha=alpha_val,
                linewidth=1.5,
                linestyle=linestyle,
                zorder=1
            )


        if len(input_view_selected_indices) > 0:
            for pred_idx in input_view_selected_indices:
                line_color = feature_colors[feat_idx]
                alpha_val = 0.9 if pred_idx in underperforming_indices else 0.8
                linewidth = 3.0
                linestyle = '-'

                label = f"#{pred_idx}" if len(input_view_selected_indices) <= 4 else None
                main_ax.plot(
                    time_steps,
                    att[pred_idx, :, feat_idx],
                    color=line_color,
                    alpha=alpha_val,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=label,
                    zorder=2
                )


        if len(input_view_selected_indices) <= 4 and feat_idx == 0:
            try:
                handles, labels = main_ax.get_legend_handles_labels()
                if handles and labels:
                    main_ax.legend(loc='upper left', fontsize=9)
            except Exception as e:
                print(f"Legend warning suppressed: {e}")


        try:
            inset_ax = inset_axes(main_ax, width="35%", height="30%", loc='upper right',
                                  bbox_to_anchor=(0.02, 0.25, 1, 1), bbox_transform=main_ax.transAxes)


            main_ax._custom_insets.append(inset_ax)

            inset_ax.grid(True, alpha=0.3)
            inset_ax.set_xlim(min_time, max_time)
            inset_ax.set_xticks(time_steps[::2])
            inset_ax.set_xticklabels(time_steps[::2], fontsize=7)
            inset_ax.tick_params(labelsize=7)
            inset_ax.set_title("Input Values", fontsize=8, pad=5)


            for pred_idx in filtered_indices:
                if pred_idx in input_view_selected_indices:
                    continue

                line_color = feature_colors[feat_idx]
                alpha_val = 0.15 if pred_idx in underperforming_indices else 0.08
                linestyle = '-'

                inset_ax.plot(
                    time_steps,
                    input_feat[pred_idx, :, feat_idx],
                    color=line_color,
                    alpha=alpha_val,
                    linewidth=0.8,
                    linestyle=linestyle,
                    zorder=1
                )


            if len(input_view_selected_indices) > 0:
                for pred_idx in input_view_selected_indices:
                    line_color = feature_colors[feat_idx]
                    alpha_val = 0.9 if pred_idx in underperforming_indices else 0.8
                    linewidth = 1.8
                    linestyle = '--' if pred_idx in underperforming_indices else '-'

                    inset_ax.plot(
                        time_steps,
                        input_feat[pred_idx, :, feat_idx],
                        color=line_color,
                        alpha=alpha_val,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        zorder=2
                    )

            print(f"DEBUG: Successfully created and tracked inset for feature {feat_idx}")

        except Exception as e:
            print(f"Warning: Could not create inset for feature {feat_idx}: {e}")

    # Summary info
    if len(input_view_selected_indices) > 0:
        underperforming_selected = len(
            [idx for idx in input_view_selected_indices if idx in underperforming_indices])
        good_selected = len(input_view_selected_indices) - underperforming_selected
        summary_text = f"{len(input_view_selected_indices)}/{len(filtered_indices)} selected"
        if underperforming_selected > 0:
            summary_text += f"\n({good_selected} good, {underperforming_selected} under)"
    else:
        underperforming_filtered = len([idx for idx in filtered_indices if idx in underperforming_indices])
        good_filtered = len(filtered_indices) - underperforming_filtered
        summary_text = f"{len(filtered_indices)} available"
        if underperforming_filtered > 0:
            summary_text += f"\n({good_filtered} good, {underperforming_filtered} under)"
        summary_text += f"\nSelect to highlight"

    try:
        ax_input_checkbox.set_title(f"Select Predictions", fontsize=11)
    except Exception as e:
        print(f"Warning: Could not update Input View checkbox title: {e}")



class FixedGridCheckButtons(FixedGridCheckButtons):
    """
    Fixed version that prevents infinite update loops
    Replace your existing FixedGridCheckButtons with this
    """

    def __init__(self, ax, num_rows, num_cols):
        super().__init__(ax, num_rows, num_cols)
        self._updating = False

    def on_click(self, event):
        """Handle click events for checkboxes - FIXED to prevent infinite loops"""


        if self._updating:
            return

        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        for i, rect in enumerate(self.checkboxes):
            xmin, ymin = rect.get_xy()
            xmax = xmin + rect.get_width()
            ymax = ymin + rect.get_height()

            if xmin <= x <= xmax and ymin <= y <= ymax:
                pred_idx = self.indices[i]

                # Set updating flag
                self._updating = True

                try:
                    self.active[pred_idx] = not self.active.get(pred_idx, False)

                    checkbox_width = rect.get_width()
                    checkbox_height = rect.get_height()
                    x_center = xmin + checkbox_width / 2
                    y_center = ymin + checkbox_height / 2

                    if self.active[pred_idx]:
                        x_mark = self.ax.text(
                            x_center, y_center, 'X',
                            ha='center', va='center',
                            fontsize=9, fontweight='bold'
                        )
                        self.x_marks[i] = x_mark
                    else:
                        if self.x_marks[i]:
                            self.x_marks[i].remove()
                            self.x_marks[i] = None


                    if current_view == 'Feature View':
                        update_feature_view()
                    elif current_view == 'Cluster View':
                        update_cluster_view()
                    elif current_view == 'Input View':
                        update_input_view()


                    fig.canvas.draw_idle()

                finally:

                    self._updating = False

                break


def on_input_view_prev_click(event):
    """Handle Previous button click in Input View - disabled for all-features view"""
    pass


def on_input_view_next_click(event):
    """Handle Next button click in Input View - disabled for all-features view"""
    pass


def on_view_change(label):
    """Enhanced view change handler with comprehensive inset cleanup and position management"""
    global feature_view_active, current_view, input_view_current_page, input_view_selected_indices, input_view_span

    current_view = label


    all_axes = ax_features + ax_predictions + ax_inputs + [ax_input_top]


    if 'cluster_axes' in globals() and cluster_axes is not None:
        all_axes.extend(cluster_axes)

    clear_all_insets_from_axes(all_axes)


    for ax in ax_features:
        if not hasattr(ax, '_original_position'):
            ax._original_position = ax.get_position()


    for ax in ax_features + ax_predictions + ax_inputs:
        ax.clear()
        ax.set_visible(False)

    ax_input_top.clear()
    ax_input_top.set_visible(False)


    ax_checkbox.set_visible(False)
    ax_input_checkbox.set_visible(False)


    if hasattr(fig, 'ax_span_selector'):
        fig.ax_span_selector.clear()
        fig.ax_span_selector.set_visible(False)

    if 'cluster_axes' in globals() and cluster_axes is not None:
        for ax in cluster_axes:
            ax.clear()
            ax.set_visible(False)


    ax_prev.set_visible(False)
    ax_next.set_visible(False)
    ax_page_info.set_visible(False)
    ax_filter.set_visible(False)
    ax_cluster.set_visible(False)
    ax_cluster_count.set_visible(False)
    ax_slider.set_visible(True)
    ax_range_text.set_visible(True)

    if label == 'Feature View':
        feature_view_active = True


        for ax in ax_features:
            if hasattr(ax, '_original_position'):
                ax.set_position(ax._original_position)
            ax.set_visible(True)

        ax_checkbox.set_visible(True)
        ax_filter.set_visible(True)
        update_feature_view()

    elif label == 'Prediction View':
        feature_view_active = False


        for ax in ax_features:
            if hasattr(ax, '_original_position'):
                ax.set_position(ax._original_position)
            ax.set_visible(False)

        for ax in ax_predictions:
            ax.set_visible(True)

        ax_prev.set_visible(True)
        ax_next.set_visible(True)
        ax_page_info.set_visible(True)
        update_prediction_view()

    elif label == 'Cluster View':
        feature_view_active = False


        for ax in ax_features:
            if hasattr(ax, '_original_position'):
                ax.set_position(ax._original_position)
            ax.set_visible(False)

        if 'cluster_axes' in globals() and cluster_axes is not None:
            for ax in cluster_axes:
                ax.set_visible(True)

        ax_prev.set_visible(True)
        ax_next.set_visible(True)
        ax_page_info.set_visible(True)
        ax_checkbox.set_visible(True)
        ax_filter.set_visible(True)
        ax_cluster.set_visible(True)
        ax_cluster_count.set_visible(True)

        if 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
            overall_clusters = cluster_results['overall']
            unique_clusters = np.array(sorted(np.unique(overall_clusters)))
            if len(unique_clusters) > 0 and current_page < len(unique_clusters):
                current_cluster = unique_clusters[current_page]
                update_cluster_range_slider(current_cluster)

        update_cluster_view(skip_slider_update=False)

    elif label == 'Input View':
        feature_view_active = False


        ax_input_checkbox.set_visible(True)
        ax_filter.set_visible(True)
        ax_all.set_visible(True)
        ax_good.set_visible(True)
        ax_bad.set_visible(True)
        ax_slider.set_visible(True)
        ax_range_text.set_visible(True)

        global input_checkbox_grid
        if input_checkbox_grid is None:
            ax_input_checkbox.clear()
            ax_input_checkbox.set_title("Select Predictions", fontsize=12)
            ax_input_checkbox.set_axis_off()
            input_checkbox_grid = FixedGridCheckButtons(ax_input_checkbox, NUM_ROWS, NUM_COLS)

        input_checkbox_grid.update_grid(filtered_indices)
        update_input_view()

    fig.canvas.draw_idle()


def on_cluster_button_click(event):

    print("Clustering button clicked")


    if current_view != 'Cluster View':
        print("Clustering is only available in Cluster View")
        return


    for ax in ax_features:
        ax.clear()
        ax.set_visible(False)


    cluster_button.label.set_text("Clustering...")
    fig.canvas.draw_idle()


    try:
        n_clusters = int(cluster_count_textbox.text)
        if n_clusters <= 0:
            print("Invalid number of clusters, using automatic detection")
            n_clusters = None
    except ValueError:
        print("Invalid number of clusters, using automatic detection")
        n_clusters = None


    results = perform_initial_clustering(n_clusters)


    cluster_button.label.set_text("Run Clustering")

    if results is None:

        if cluster_axes is not None and len(cluster_axes) > 0:
            cluster_axes[0].clear()
            cluster_axes[0].set_visible(True)
            cluster_axes[0].text(0.5, 0.5, "Clustering failed. Check console for errors.",
                                ha='center', va='center', fontsize=12, color='red',
                                transform=cluster_axes[0].transAxes)
            fig.canvas.draw_idle()
    else:

        if 'overall' in results:
            num_clusters = len(np.unique(results['overall']))
            page_text.set_text(f"Cluster View - {num_clusters} clusters found")


        update_cluster_view(skip_slider_update=False)

    fig.canvas.draw_idle()


def on_prev_click(event):
    global current_page, input_view_current_page

    if current_view == 'Input View':
        if input_view_current_page > 0:
            input_view_current_page -= 1
            update_input_view()
    elif current_page > 0:
        current_page -= 1
        if current_view == 'Prediction View':
            update_prediction_view()
        elif current_view == 'Cluster View':
            if 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
                overall_clusters = cluster_results['overall']
                unique_clusters = np.array(sorted(np.unique(overall_clusters)))
                if len(unique_clusters) > 0 and current_page < len(unique_clusters):
                    current_cluster = unique_clusters[current_page]
                    update_cluster_range_slider(current_cluster)
            update_cluster_view()

    fig.canvas.draw_idle()


def on_next_click(event):
    global current_page, input_view_current_page

    if current_view == 'Input View':
        predictions_per_page = 6
        total_pages = max(1, (len(input_view_selected_indices) + predictions_per_page - 1) // predictions_per_page)
        if input_view_current_page < total_pages - 1:
            input_view_current_page += 1
            update_input_view()
    elif current_view == 'Prediction View':
        display_preds = filtered_indices
        total_pages = max(1, (len(display_preds) + plots_per_page - 1) // plots_per_page)
        if current_page < total_pages - 1:
            current_page += 1
            update_prediction_view()
    elif current_view == 'Cluster View':
        if 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
            unique_clusters = np.unique(cluster_results['overall'])
            total_pages = len(unique_clusters)
            if current_page < total_pages - 1:
                current_page += 1
                unique_clusters = np.array(sorted(np.unique(cluster_results['overall'])))
                if len(unique_clusters) > 0 and current_page < len(unique_clusters):
                    current_cluster = unique_clusters[current_page]
                    update_cluster_range_slider(current_cluster)
                update_cluster_view()

    fig.canvas.draw_idle()


def hover(event):

    if event.inaxes != ax_scatter:
        tooltip.set_visible(False)
        fig.canvas.draw_idle()
        return


    if scatter_pred is not None:

        cont_pred, ind_pred = scatter_pred.contains(event)

        if cont_pred:

            ind = ind_pred['ind'][0]
            idx = filtered_indices[ind]
            value = pred[idx]

            #
            tooltip.xy = (idx, value)


            target_value = targets[idx]
            error = value - target_value
            is_bad = "Yes" if idx in underperforming_indices else "No"


            tooltip_text = f"Index: {idx}\nPrediction: {value:.4f}\nTarget: {target_value:.4f}" + \
                           f"\nError: {error:.4f}\nUnderperforming: {is_bad}"


            if current_view == 'Cluster View' and 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
                cluster_id = cluster_results['overall'][idx]
                cluster_info = f"\nCluster: {cluster_id}"


                if 'exemplars' in cluster_results:
                    exemplars = cluster_results['exemplars']
                    if idx in exemplars.values():
                        cluster_info += " (Exemplar)"


                if 'centroid_distances' in cluster_results and idx in cluster_results['centroid_distances']:
                    distance = cluster_results['centroid_distances'][idx]

                    max_dist = max(cluster_results['centroid_distances'].values())
                    typicality = max(0, 100 * (1 - distance / max_dist))
                    cluster_info += f"\nTypicality: {typicality:.1f}%"

                tooltip_text += cluster_info

            tooltip.set_text(tooltip_text)


            if idx in underperforming_indices:
                tooltip.get_bbox_patch().set_facecolor('red')
                tooltip.get_bbox_patch().set_alpha(0.2)
            else:
                tooltip.get_bbox_patch().set_facecolor('blue')
                tooltip.get_bbox_patch().set_alpha(0.2)


            tooltip.set_visible(True)
        else:

            tooltip.set_visible(False)

        fig.canvas.draw_idle()


def on_pick(event):

    if event.mouseevent.inaxes != ax_scatter:
        return


    if event.artist == scatter_pred:
        ind = event.ind[0]
        idx = filtered_indices[ind]
        value = pred[idx]


        tooltip.xy = (idx, value)


        target_value = targets[idx]
        error = value - target_value
        is_bad = "Yes" if idx in underperforming_indices else "No"


        tooltip_text = f"Index: {idx}\nPrediction: {value:.4f}\nTarget: {target_value:.4f}" + \
                       f"\nError: {error:.4f}\nUnderperforming: {is_bad}"


        if current_view == 'Cluster View' and 'cluster_results' in globals() and cluster_results is not None and 'overall' in cluster_results:
            cluster_id = cluster_results['overall'][idx]
            cluster_info = f"\nCluster: {cluster_id}"


            if 'exemplars' in cluster_results:
                exemplars = cluster_results['exemplars']
                if idx in exemplars.values():
                    cluster_info += " (Exemplar)"


            if 'centroid_distances' in cluster_results and idx in cluster_results['centroid_distances']:
                distance = cluster_results['centroid_distances'][idx]

                max_dist = max(cluster_results['centroid_distances'].values())
                typicality = max(0, 100 * (1 - distance / max_dist))
                cluster_info += f"\nTypicality: {typicality:.1f}%"

            tooltip_text += cluster_info

        tooltip.set_text(tooltip_text)


        if idx in underperforming_indices:
            tooltip.get_bbox_patch().set_facecolor('red')
            tooltip.get_bbox_patch().set_alpha(0.2)
        else:
            tooltip.get_bbox_patch().set_facecolor('blue')
            tooltip.get_bbox_patch().set_alpha(0.2)


        tooltip.set_visible(True)

        fig.canvas.draw_idle()


def on_all_button(event):

    global current_perf_filter
    current_perf_filter = 'all'
    update_filtered_predictions(None)


def on_good_button(event):

    global current_perf_filter
    current_perf_filter = 'good'
    update_filtered_predictions(None)


def on_bad_button(event):

    global current_perf_filter
    current_perf_filter = 'bad'
    update_filtered_predictions(None)



def on_slider_change(val):


    if is_updating_slider:
        return


    current_view_before = current_view


    update_filtered_predictions(val)


    ax_slider.set_xticks([])
    ax_slider.set_yticks([])
    ax_slider.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)


    for txt in plt.findobj(plt.Text):
        if hasattr(txt, 'get_position'):
            pos_x, pos_y = txt.get_position()

            if (0.1 <= pos_x <= 0.9) and (0.18 <= pos_y <= 0.22) and txt != range_text:
                txt.set_visible(False)

    fig.canvas.draw_idle()



def on_filter_click(event):

    if checkbox_grid is not None:
        checkbox_grid.clear_all_selections()


    fig.canvas.draw_idle()



def cluster_feature_patterns(attributions, n_clusters=None, method='overall'):


    global cluster_results

    try:

        if attributions is None or not isinstance(attributions, np.ndarray):
            print("Error: Invalid attributions data")
            return None, None

        print(f"Attribution shape: {attributions.shape}")
        print(f"Using n_clusters: {n_clusters}")


        flattened_full_attrs = attributions.reshape(attributions.shape[0], -1)
        print(f"Flattened shape: {flattened_full_attrs.shape}")


        if np.isnan(flattened_full_attrs).any() or np.isinf(flattened_full_attrs).any():
            print("Warning: Data contains NaN or Inf values. Replacing with zeros.")
            flattened_full_attrs = np.nan_to_num(flattened_full_attrs)

        # Standardize
        print("Standardizing data...")
        full_scaler = StandardScaler()
        scaled_full_attrs = full_scaler.fit_transform(flattened_full_attrs)


        from sklearn.cluster import AgglomerativeClustering

        print("Running clustering...")


        if n_clusters is not None and n_clusters > 0:
            print(f"Using specified number of clusters: {n_clusters}")
            overall_clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='cosine',
                linkage='average'
            )
        else:

            try:
                overall_clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.5,
                    affinity='cosine',
                    linkage='average'
                )
            except Exception as e:
                print(f"Error with automatic clustering: {str(e)}")
                print("Falling back to fixed number of clusters...")
                overall_clusterer = AgglomerativeClustering(
                    n_clusters=min(5, attributions.shape[0] // 2),
                    affinity='cosine',
                    linkage='average'
                )


        overall_clusters = overall_clusterer.fit_predict(scaled_full_attrs)


        if len(np.unique(overall_clusters)) <= 1 and n_clusters is None:
            print("Only one cluster found, trying with fixed number...")
            overall_clusterer = AgglomerativeClustering(
                n_clusters=min(5, attributions.shape[0] // 2),
                affinity='cosine',
                linkage='average'
            )
            overall_clusters = overall_clusterer.fit_predict(scaled_full_attrs)


        unique, counts = np.unique(overall_clusters, return_counts=True)
        print("\nOverall Clustering:")
        for cluster, count in zip(unique, counts):
            print(f"Cluster {cluster}: {count} predictions ({count / len(overall_clusters) * 100:.2f}%)")


        try:

            db_score = davies_bouldin_score(scaled_full_attrs, overall_clusters)
            print(f"\nDavies-Bouldin Index: {db_score:.4f}")
        except Exception as db_error:
            print(f"Error calculating Davies-Bouldin Index: {db_error}")
            db_score = None


        cluster_results = {
            'overall': overall_clusters,
            'davies_bouldin_score': db_score
        }

        print("Calculating centroids and exemplars...")

        centroids = {}
        for cluster_id in np.unique(overall_clusters):
            cluster_mask = overall_clusters == cluster_id
            centroid = np.mean(scaled_full_attrs[cluster_mask], axis=0)
            centroids[cluster_id] = centroid


        from scipy.spatial.distance import euclidean
        exemplars = {}
        centroid_distances = {}

        for pred_idx in range(len(overall_clusters)):
            cluster_id = overall_clusters[pred_idx]
            centroid = centroids[cluster_id]
            distance = euclidean(scaled_full_attrs[pred_idx], centroid)
            centroid_distances[pred_idx] = distance

        for cluster_id in np.unique(overall_clusters):
            cluster_indices = np.where(overall_clusters == cluster_id)[0]
            distances = [centroid_distances[idx] for idx in cluster_indices]
            exemplar_idx = cluster_indices[np.argmin(distances)]
            exemplars[cluster_id] = exemplar_idx

        print("\nCluster exemplars (representative predictions):")
        for cluster_id, exemplar_idx in exemplars.items():
            print(f"Cluster {cluster_id}: Prediction #{exemplar_idx}")


        cluster_results['exemplars'] = exemplars
        cluster_results['centroid_distances'] = centroid_distances

        print("Clustering completed successfully!")
        return overall_clusters, cluster_results

    except Exception as e:
        import traceback
        print(f"Error in clustering: {str(e)}")
        traceback.print_exc()
        return None, None



def perform_initial_clustering(n_clusters=None):

    global cluster_results, current_page


    if current_view != 'Cluster View':
        print("Clustering is only available in Cluster View")
        return None

    print(f"Performing clustering with specified clusters: {n_clusters}...")
    print(f"Input attribution data shape: {att.shape}")


    if att is None or not isinstance(att, np.ndarray):
        print("Error: Attribution data is invalid")
        return None


    att_copy = np.array(att)
    if np.isnan(att_copy).any() or np.isinf(att_copy).any():
        print("Warning: Data contains NaN or Inf values. Replacing with zeros.")
        att_copy = np.nan_to_num(att_copy)


    current_page = 0


    overall_clusters, results = cluster_feature_patterns(att_copy, n_clusters=n_clusters)

    if overall_clusters is None or results is None:
        print("Clustering failed. Check earlier error messages.")
        return None


    cluster_results = {}
    cluster_results['overall'] = results['overall']


    if 'davies_bouldin_score' in results:
        cluster_results['davies_bouldin_score'] = results['davies_bouldin_score']
        print(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")

        if hasattr(fig, 'db_score_text'):
            fig.db_score_text.set_text(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
            fig.db_score_ax.set_visible(True)
        else:

            db_ax = plt.axes([0.7, 0.01, 0.25, 0.02])
            db_ax.set_axis_off()
            fig.db_score_text = db_ax.text(0, 0.5, f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}",
                                           fontsize=8, color='gray')
            fig.db_score_ax = db_ax


    if 'centroid_distances' in results:
        cluster_results['centroid_distances'] = results['centroid_distances']

    print("Clustering complete! Updating visualization...")

    try:

        update_scatter_plot(filtered_indices)


        if current_view == 'Cluster View':

            update_cluster_view(skip_slider_update=False)

        fig.canvas.draw_idle()
        print("Visualization updated successfully")
    except Exception as e:
        import traceback
        print(f"Error updating visualization: {str(e)}")
        traceback.print_exc()

    return cluster_results


range_slider.on_changed(on_slider_change)
filter_button.on_clicked(on_filter_click)
view_radio.on_clicked(on_view_change)


all_button.on_clicked(on_all_button)
good_button.on_clicked(on_good_button)
bad_button.on_clicked(on_bad_button)


prev_button.on_clicked(on_prev_click)
next_button.on_clicked(on_next_click)


cluster_button.on_clicked(on_cluster_button_click)


fig.canvas.mpl_connect('motion_notify_event', hover)
fig.canvas.mpl_connect('pick_event', on_pick)

if not hasattr(fig, 'ax_span_selector'):

    fig.ax_span_selector = fig.add_axes([0.1, 0.22, 0.8, 0.15])
    fig.ax_span_selector.set_visible(False)

update_scatter_plot(np.arange(num_predictions))


update_filtered_predictions((min_pred, max_pred))


if current_view == 'Input View':

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
else:
    # Original layout for other views
    plt.tight_layout(rect=[0, 0, 1, 0.97])

def on_figure_resize(event):
    """Keep buttons visible after figure resize for Input View"""
    if current_view == 'Input View' and 'input_view_ax_prev' in globals() and input_view_ax_prev is not None:
        if len(input_view_selected_indices) > 0:
            # Ensure buttons remain visible after resize
            input_view_ax_prev.set_visible(True)
            input_view_ax_next.set_visible(True)
            input_view_ax_page_info.set_visible(True)
            fig.canvas.draw_idle()


fig.canvas.mpl_connect('resize_event', on_figure_resize)


plt.show()
