"""
Utility functions for plotting
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib import rc
from sklearn.calibration import calibration_curve


CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00',
				  '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
HATCHINGS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
MARKERS = ['D', 'o', '^', 'v', 's', 'X', '*', 'D', 'o', '^', 'v', 's', 'X', '*', 'D', 'o', '^', 'v', 's', 'X', '*']
STYLES = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1, 1))]


def plotting_setup(font_size=12):
	"""
	Sets global plot formatting settings
	"""
	plt.style.use("seaborn-colorblind")
	plt.rcParams['font.size'] = font_size
	rc('text', usetex=False)
	plt.rcParams["font.family"] = "sans-serif"
	rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})


def plot_curves_with_ci_lu(xs, avgs, lower, upper, labels, xlab, ylab, font_size=16, title=None, baseline=None,
                           baseline_lab=None, baseline_cl=None, dir=None, legend=True, legend_outside=True,
                           cls=None, ms=None, markersize=16, linewidth=5, figsize=(10, 5), ylim=None, tick_step=None,
						   grid=False):
	"""
	Plots several curves with the given confidence bands
	"""
	plotting_setup(font_size)

	fig = plt.figure(figsize=figsize)

	if cls is None:
		cls = CB_COLOR_CYCLE[:len(xs)]
	if ms is None:
		ms = MARKERS[:len(xs)]

	if baseline_cl is None:
		baseline_cl = 'red'

	if baseline is not None:
		plt.axhline(baseline, label=baseline_lab, c=baseline_cl, linestyle='--')

	for i in range(len(xs)):
		upper_i = upper[i]
		lower_i = lower[i]

		plt.plot(xs[i], avgs[i], color=cls[i], label=labels[i], marker=ms[i], markersize=markersize,
				 linewidth=linewidth, linestyle=STYLES[i])
		plt.fill_between(xs[i], lower_i, upper_i, color=cls[i], alpha=0.1)

	plt.xlabel(xlab)
	plt.ylabel(ylab)

	if title is not None:
		plt.title(title)

	if ylim is not None:
		plt.ylim(ylim)

	if tick_step is not None and ylim is not None:
		y_min = ylim[0]
		y_max = ylim[1]
		yticks = np.arange(y_min, y_max, tick_step)
		plt.yticks(ticks=yticks)

	if grid:
		plt.grid(visible=True, axis='y')

	if legend:
		if legend_outside:
			leg = plt.legend(loc='lower center', ncol=len(xs), bbox_to_anchor=(0.5, -0.3), frameon=False)
		else:
			leg = plt.legend(loc='upper right', frameon=False)

		# Change the marker size manually for both lines
		for i in range(len(leg.legendHandles)):
			leg.legendHandles[i].set_markersize(16)
			leg.legendHandles[i].set_linewidth(5.0)

	if dir is not None:
		plt.savefig(fname=dir, dpi=300, bbox_inches='tight')
	else:
		plt.show()


def plot_calibration_curves(y_scores, y_true, labels=None, cls=None, n_bins=10, linewidth=5, markersize=16,
							font_size=16, dir=None, ms=None, stl=None, legend=True, legend_outside=False):
	"""
		Plots calibration curves
	"""
	plotting_setup(font_size)

	fig, ax = plt.subplots(figsize=(10, 10))

	if cls is None:
		cls = CB_COLOR_CYCLE
	if ms is None:
		ms = MARKERS
	if stl is None:
		stl = STYLES

	for j in range(len(y_scores)):
		cal_y, cal_x = calibration_curve(y_true, y_scores[j], n_bins=n_bins)

		# Plot the calibration curve
		if labels is None:
			plt.plot(cal_x, cal_y, marker=ms[j], c=cls[j], linewidth=linewidth, linestyle=stl[j],  markersize=markersize)
		else:
			plt.plot(cal_x, cal_y, marker=ms[j], c=cls[j], linewidth=linewidth,  linestyle=stl[j], markersize=markersize,
					 label=labels[j])

	# reference line, legends, and axis labels
	line = mlines.Line2D([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect')
	transform = ax.transAxes
	line.set_transform(transform)
	ax.add_line(line)
	ax.set_xlabel('Predicted Probability')
	ax.set_ylabel('True Probability')
	plt.grid(visible=True, axis='y')
	plt.grid(visible=True, axis='x')
	plt.xlim([0, 1])
	plt.ylim([0, 1])

	if legend:
		if legend_outside:
			leg = plt.legend(loc='lower center', ncol=len(y_scores) + 1, bbox_to_anchor=(0.5, -0.3), frameon=False)
		else:
			leg = plt.legend(loc='upper right', frameon=False)

		# Change the marker size manually for both lines
		for i in range(len(leg.legendHandles)):
			leg.legendHandles[i].set_markersize(16)
			leg.legendHandles[i].set_linewidth(5.0)

	if dir is not None:
		plt.savefig(fname=dir, dpi=300, bbox_inches='tight')
	else:
		plt.show()


def plot_calibration_curves_multiclass(y_scores, y_true, classes, labels=None, cls=None, n_bins=10, linewidth=5,
									   markersize=16, font_size=16, alpha=0.5, dir=None):
	"""
		Plots calibration curves for multiclass settings
		NOTE: @classes is a list/array of class labels/indices for which calibration curves are to be plotted
	    NOTE: introduced transparency for calibration curves for better readability
	"""
	#
	plotting_setup(font_size)

	fig, ax = plt.subplots(figsize=(10, 10))

	if cls is None:
		cls = CB_COLOR_CYCLE

	for j in range(len(y_scores)):

		cnt = 0

		for c in classes:
			cal_y, cal_x = calibration_curve((y_true == c) * 1, y_scores[j][:, c], n_bins=n_bins)

			# Plot the calibration curve
			if labels is None or cnt >= 1:
				plt.plot(cal_x, cal_y, marker=MARKERS[j], c=cls[j], linewidth=linewidth, markersize=markersize,
						 alpha=alpha)
			else:
				plt.plot(cal_x, cal_y, marker=MARKERS[j], c=cls[j], linewidth=linewidth, markersize=markersize,
						 label=labels[j], alpha=alpha)

			cnt += 1

	# reference line, legends, and axis labels
	line = mlines.Line2D([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect')
	transform = ax.transAxes
	line.set_transform(transform)
	ax.add_line(line)
	ax.set_xlabel('Predicted Probability')
	ax.set_ylabel('True Probability')
	plt.grid(visible=True, axis='y')
	plt.grid(visible=True, axis='x')
	plt.xlim([0, 1])
	plt.ylim([0, 1])

	leg = plt.legend()

	# Change the marker size manually for both lines
	for i in range(len(leg.legendHandles)):
		leg.legendHandles[i].set_markersize(16)
		leg.legendHandles[i].set_linewidth(5.0)

	if dir is not None:
		plt.savefig(fname=dir, dpi=300, bbox_inches='tight')
	else:
		plt.show()
