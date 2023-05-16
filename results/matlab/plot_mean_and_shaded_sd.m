function plot_mean_and_shaded_sd(x, y, name)
    ax = gca;
    ax.LineStyleOrderIndex = ax.ColorOrderIndex;
    order_index = ax.ColorOrderIndex;
    line_style_order = ax.LineStyleOrder;
    
    tbl = table(x, y);
    tblstats = grpstats(tbl, "x", ["mean", "std"]);
    x_new = tblstats.x;
    mean_y = tblstats.mean_y;

    mean_line = plot(x_new, mean_y, string(ax.LineStyleOrder(ax.LineStyleOrderIndex)), 'DisplayName', name(1), 'MarkerSize', 4);
    hold on

    std_above = tblstats.mean_y - tblstats.std_y;
    std_below = tblstats.mean_y + tblstats.std_y;

    xconf = [x_new; x_new(end:-1:1)];
    yconf = [std_above; std_below(end:-1:1)];
    color = get(mean_line, 'Color');
    p = fill(xconf, yconf, color, 'HandleVisibility', 'off');
    alpha(p, .2);
%     p.FaceColor = [1 0.8 0.8];
    p.EdgeColor = 'none';
    
    ax.LineStyleOrder = line_style_order;
    ax.ColorOrderIndex = order_index + 1;
    ax.LineStyleOrderIndex = order_index + 1;
end