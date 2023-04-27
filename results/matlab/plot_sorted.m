function plot_sorted(x, y, name)
    [xsorted, I] = sort(x);
    ysorted = y(I);
    plot(xsorted, ysorted, 'DisplayName', name(1), 'MarkerSize', 4);
end