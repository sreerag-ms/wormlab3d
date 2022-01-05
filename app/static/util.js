$(document).ready(function() {
    // Remember active tab
    $('button[data-bs-toggle="tab"]').on('shown.bs.tab', function (e) {
        let tabs_id = $(this).closest('ul.nav-tabs').attr('id');
        console.log('setting ', 'activeTab_' + tabs_id, $(e.target).data('bs-target'));
        localStorage.setItem('activeTab_' + tabs_id, $(e.target).data('bs-target'));

        // Resize data tables
        $($.fn.dataTable.tables(true)).DataTable().columns.adjust();
    });

    // Restore active tab
    $('ul.nav-tabs').each(function(el) {
        let tabs_id = $(this).attr('id');
        let activeTab = localStorage.getItem('activeTab_' + tabs_id);
        if (activeTab) {
            $('button[data-bs-target="' + activeTab + '"]').tab('show');
        }
        console.log('restoring ', 'activeTab_' + tabs_id, activeTab);
    });

    // Re-layout plots when tab is first shown
    $('button[data-bs-toggle="tab"]').one('shown.bs.tab', function () {
        $('.js-plotly-plot', $(this).data('bs-target')).one('plotly_afterplot', function() {
            Plotly.relayout(this, {autosize: true});
        });
    });
});


// ================================= Tracking plots =================================

let plot_config = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['toImage']
};

function timestamps_in_ms(timestamps) {
    return timestamps.map(function (t) {
        return t * 1000
    });
}

function plot_tracking_single(X, idx, timestamps) {
    let xyz = ['x', 'y', 'z'][idx];
    let data = [{
        x: timestamps_in_ms(timestamps),
        y: X[idx],
        type: 'scattergl',
        mode: 'markers',
        marker: {
            size: 2,
            color: timestamps,
            colorscale: 'Jet'
        },
    }];
    let layout = {
        title: xyz,
        margin: {t: 50},
        xaxis: {
            title: 'Time (s)',
            type: 'date',
            tickformat: '%M:%S'
        },
        yaxis: {
            title: 'Position (mm)',
        }
    };

    Plotly.newPlot('tracking-plot-' + xyz, data, layout, plot_config);
}

function plot_tracking_pairs(X, idx0, idx1, timestamps) {
    let xyz0 = ['x', 'y', 'z'][idx0];
    let xyz1 = ['x', 'y', 'z'][idx1];
    let pair = xyz0 + xyz1;
    let data = [{
        x: X[idx0],
        y: X[idx1],
        type: 'scattergl',
        mode: 'markers',
        marker: {
            size: 2,
            color: timestamps,
            colorscale: 'Jet'
        },
    }];
    let layout = {
        title: pair,
        margin: {t: 50},
        xaxis: {
            title: xyz0,
        },
        yaxis: {
            title: xyz1
        }
    };
    Plotly.newPlot('tracking-plot-' + pair, data, layout, plot_config);
}

function plot_tracking_3d(X, timestamps) {
    let data = [{
        x: X[0],
        y: X[1],
        z: X[2],
        type: 'scatter3d',
        mode: 'markers',
        marker: {
            size: 2,
            color: timestamps,
            colorscale: 'Jet'
        },
    }];
    let layout = {
        margin: {t: 50},
        xaxis: {
            title: 'x',
        },
        yaxis: {
            title: 'y'
        },
        zaxis: {
            title: 'z'
        }
    };
    Plotly.newPlot('tracking-plot-3d', data, layout, plot_config);
}

// ==================================================================

