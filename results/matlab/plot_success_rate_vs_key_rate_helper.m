function plot_success_rate_vs_key_rate_helper(code_strategy, q, A, N, p_err, max_qkd_leak, ax)
    [B, G, group_names] = filter_and_group_data(code_strategy, A);

    % f_kr = figure;
    % f_kr.Units = 'inches';
    % f_kr.Position(3:4) = 1.2*[4.5 2.4];
    % ax = axes();
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    xline(ax, B.theoretic_key_rate(1), 'DisplayName', 'Shannon Limit','LineWidth', 1.5, 'Color', 'b');
    % yline(log2(q)-Bridge2D.qd ./ Bridge2D.sift_vec, 'DisplayName', 'Bridge[2,3] Minimum','LineWidth', 1.5);
    splitapply(@plot_sorted, B.key_rate_completed_only(:), B.is_success(:), group_names(:), G(:))
    % splitapply(plot_ir, B.key_rate_success_only(:), B.ser_b_key_completed_only(:), group_names(:), G(:))
    if (p_err > 0.0) && ismember(q, [3, 5])
        xline(ax, log2(q)-max_qkd_leak, 'DisplayName', sprintf('Bridge[%d,%d] Minimum', q-1, q),'LineWidth', 1.5, 'Color', 'r');
    end
    xlabel(ax, "Key Rate, $R$ [secret bits per sifted pulse]",'Interpreter','latex','FontSize', 9) 
    ylabel(ax, "Success Rate, $\alpha$",'Interpreter','latex','FontSize', 9);
    title(ax, sprintf("Success Rate vs Key Rate, N=%d", N))
    % axis padded
    box on
    legend('boxoff');
    legend('Interpreter','latex','FontSize', 8);
    legend('Location','eastoutside');

    % f_kr.CurrentAxes.TickLabelInterpreter = 'latex';