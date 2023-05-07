function plots_ir(code_strategy, q, Q, success_rate, save)
    p_err = str2double(Q);
    [A, max_qkd_leak] = get_data(code_strategy, q, Q, false);
    if strcmp(code_strategy, 'mb')
        tol = 5 * eps(100); % A very small value
        A = A(ismembertol(A.mb_desired_success_rate(:), success_rate, tol), :);
    end
    A = A(A.is_success == "True", :);
    [B, G, group_names] = filter_and_group_data(code_strategy, A);

    f_kr = figure;
    f_kr.Units = 'inches';
    f_kr.Position(3:4) = 1.2*[4.5 2.4];
    ax = axes(); 
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    hold on
    yline(B.theoretic_key_rate(1), 'DisplayName', 'Shannon Limit','LineWidth', 1.5, 'Color', 'b');
    % yline(log2(q)-Bridge2D.qd ./ Bridge2D.sift_vec, 'DisplayName', 'Bridge[2,3] Minimum','LineWidth', 1.5);
    % splitapply(plot_ir, B.N(:), B.key_rate_success_only(:), group_names(:), G(:))
    splitapply(@plot_mean_and_shaded_sd, B.N(:), B.key_rate_success_only(:), group_names(:), G(:))
    if (p_err > 0.0) && ismember(q, [3, 5])
        yline(log2(q)-max_qkd_leak, 'DisplayName', sprintf('Bridge[%d,%d] Minimum', q-1, q),'LineWidth', 1.5, 'Color', 'r');
    end
    xlabel("Key Length, $N$",'Interpreter','latex','FontSize', 9);
    ylabel("Key Rate, $R$ [secret bits per sifted pulse]",'Interpreter','latex','FontSize', 9) 
    axis padded
    box on
    legend('boxoff');
    legend('Interpreter','latex','FontSize', 8);
    legend('Location','eastoutside');
    f_kr.CurrentAxes.TickLabelInterpreter = 'latex';
    hold off
    
    f_tr = figure;
    f_tr.Units = 'inches';
    f_tr.Position(3:4) = 1.2*[4.5 2.4];
    ax = axes(); 
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    hold on
    splitapply(@plot_mean_and_shaded_sd, B.N(:), B.time_rate(:), group_names(:), G(:))
    xlabel("Key Length, $N$",'Interpreter','latex','FontSize', 9);
    ylabel("Time Rate [seconds per sifted pulse]",'Interpreter','latex','FontSize', 9);
    axis padded
    box on
    legend('boxoff');
    legend('Interpreter','latex','FontSize', 8);
    legend('Location','eastoutside');
    f_tr.CurrentAxes.TickLabelInterpreter = 'latex';
    legend
    hold off
    
    f_se = figure;
    f_se.Units = 'inches';
    f_se.Position(3:4) = 1.2*[4.5 2.4];
    ax = axes(); 
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    gap = B.theoretic_key_rate - B.key_rate_success_only;
    scaling_exponent = -log(B.N) ./ log(gap);
    hold on
    splitapply(@plot_mean_and_shaded_sd, B.N(:), scaling_exponent(:), group_names(:), G(:))
    xlabel("Key Length, $N$",'Interpreter','latex','FontSize', 9);
    ylabel("Scaling Exponent, $\rho$",'Interpreter','latex','FontSize', 9);
    axis padded
    box on
    legend('boxoff');
    legend('Interpreter','latex','FontSize', 8);
    legend('Location','eastoutside');
    f_se.CurrentAxes.TickLabelInterpreter = 'latex';
    legend
    hold off

    if save
        saveas(f_kr, sprintf('PycharmProjects/qsc_ir/results/matlab/plots/key_rate,%s,q=%d,Q=%f,success_rate=%f.svg', code_strategy, q, p_err, success_rate));
        saveas(f_kr, sprintf('PycharmProjects/qsc_ir/results/matlab/plots/key_rate,%s,q=%d,Q=%f,success_rate=%f.png', code_strategy, q, p_err, success_rate));
        saveas(f_tr, sprintf('PycharmProjects/qsc_ir/results/matlab/plots/time_rate,%s,q=%d,Q=%f,success_rate=%f.svg', code_strategy, q, p_err, success_rate));
        saveas(f_tr, sprintf('PycharmProjects/qsc_ir/results/matlab/plots/time_rate,%s,q=%d,Q=%f,success_rate=%f.png', code_strategy, q, p_err, success_rate));
        saveas(f_se, sprintf('PycharmProjects/qsc_ir/results/matlab/plots/scaling_exponent,%s,q=%d,Q=%f,success_rate=%f.svg', code_strategy, q, p_err, success_rate));
        saveas(f_se, sprintf('PycharmProjects/qsc_ir/results/matlab/plots/scaling_exponent,%s,q=%d,Q=%f,success_rate=%f.png', code_strategy, q, p_err, success_rate));
    end
end