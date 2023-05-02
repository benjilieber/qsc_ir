function ir_single_plot(code_strategy, q, p_err, Q, success_rate, alpha, block_length, list_size, max_num_indices_to_encode, save)
    [A, max_qkd_leak] = get_data(code_strategy, q, Q, false);
    tol = 5 * eps(100); % A very small value
    A = A(ismembertol(A.p_err, p_err, tol) & ...
        ismembertol(A.mb_desired_success_rate, success_rate, tol), :);
    A = A(ismember(A.result_type, 'full_reduce') & ...
        ismembertol(A.mb_block_length(:), block_length, tol) & ...
        ismembertol(A.list_size(:), list_size, tol) & ...
        ismembertol(A.mb_max_num_indices_to_encode(:), max_num_indices_to_encode, tol) & ...
        ismembertol(A.mb_desired_success_rate(:), success_rate, tol), :);
    A = A(A.is_success == "True", :);
    group_names = string(A.mb_block_length) + ',' + string(A.list_size) + ',' + string(A.mb_max_num_indices_to_encode) + '-mb';
    
    f_kr = figure;
    f_kr.Units = 'inches';
    f_kr.Position(3:4) = 1.2*[4.5 2.4];
    ax = axes(); 
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    hold on
    yline(A.theoretic_key_rate(1), 'DisplayName', 'Shannon Limit','LineWidth', 1.5, 'Color', 'b');
    plot_mean_and_shaded_sd(A.N(:), A.key_rate_success_only(:), group_names(:))
    if (p_err > 0.0) && ismember(q, [3, 5])
        yline(log2(q)-max_qkd_leak, 'DisplayName', sprintf('Bridge[%d,%d] Minimum', q-1, q),'LineWidth', 1.5, 'Color', 'r');
    end
    xlabel("Key Length, $N$",'Interpreter','latex','FontSize', 9);
    ylabel("Key Rate, $R$ [secret bits per sifted pulse]",'Interpreter','latex','FontSize', 9) 
    axis padded
    box on
    % legend('Interpreter','latex','FontSize', 8);
    % legend('Location','southeast');
    % f_kr.CurrentAxes.TickLabelInterpreter = 'latex';
    hold off
    
    f_tr = figure;
    f_tr.Units = 'inches';
    f_tr.Position(3:4) = 1.2*[4.5 2.4];
    ax = axes(); 
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    hold on
    plot_mean_and_shaded_sd(A.N(:), A.time_rate(:), group_names(:));
    xlabel("Key Length, $N$",'Interpreter','latex','FontSize', 9);
    ylabel("Time Rate [seconds per sifted pulse]",'Interpreter','latex','FontSize', 9);
    axis padded
    box on
    f_tr.CurrentAxes.TickLabelInterpreter = 'latex';
    hold off
    
    f_ctr = figure;
    f_ctr.Units = 'inches';
    f_ctr.Position(3:4) = 1.2*[4.5 2.4];
    ax = axes(); 
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    hold on
    plot_mean_and_shaded_sd(A.N(:), A.cpu_time_rate(:), group_names(:));
    xlabel("Key Length, $N$",'Interpreter','latex','FontSize', 9);
    ylabel("CPU Time Rate [seconds per sifted pulse]",'Interpreter','latex','FontSize', 9);
    axis padded
    box on
    f_ctr.CurrentAxes.TickLabelInterpreter = 'latex';
    hold off
        
    f_se = figure;
    f_se.Units = 'inches';
    f_se.Position(3:4) = 1.2*[4.5 2.4];
    ax = axes(); 
    ax.LineStyleOrder = {'--x','--o', '--^', '--*', '-x','-o', '-^', '-*'};
    gap = A.theoretic_key_rate - A.key_rate_success_only;
    scaling_exponent = -log(A.N) ./ log(gap);
    hold on
    plot_mean_and_shaded_sd(A.N(:), scaling_exponent(:), group_names(:))
    xlabel("Key Length, $N$",'Interpreter','latex','FontSize', 9);
    ylabel("Scaling Exponent, $\rho$",'Interpreter','latex','FontSize', 9);
    axis padded
    box on
    f_se.CurrentAxes.TickLabelInterpreter = 'latex';
    hold off
    
    if save
        saveas(f_kr, sprintf('/cs/usr/benjilieber/PycharmProjects/qsc_ir/results/matlab/plots/key_rate,%s,q=%d,Q=%s,success_rate=%s,n=%d,L=%d,#=%d.svg', code_strategy, q, Q, alpha, block_length, list_size, max_num_indices_to_encode));
        saveas(f_kr, sprintf('/cs/usr/benjilieber/PycharmProjects/qsc_ir/results/matlab/plots/key_rate,%s,q=%d,Q=%s,success_rate=%s,n=%d,L=%d,#=%d.png', code_strategy, q, Q, alpha, block_length, list_size, max_num_indices_to_encode));
        saveas(f_tr, sprintf('/cs/usr/benjilieber/PycharmProjects/qsc_ir/results/matlab/plots/time_rate,%s,q=%d,Q=%s,success_rate=%s,n=%d,L=%d,#=%d.svg', code_strategy, q, Q, alpha, block_length, list_size, max_num_indices_to_encode));
        saveas(f_tr, sprintf('/cs/usr/benjilieber/PycharmProjects/qsc_ir/results/matlab/plots/time_rate,%s,q=%d,Q=%s,success_rate=%s,n=%d,L=%d,#=%d.png', code_strategy, q, Q, alpha, block_length, list_size, max_num_indices_to_encode));
        saveas(f_ctr, sprintf('/cs/usr/benjilieber/PycharmProjects/qsc_ir/results/matlab/plots/cpu_time_rate,%s,q=%d,Q=%s,success_rate=%s,n=%d,L=%d,#=%d.svg', code_strategy, q, Q, alpha, block_length, list_size, max_num_indices_to_encode));
        saveas(f_ctr, sprintf('/cs/usr/benjilieber/PycharmProjects/qsc_ir/results/matlab/plots/cpu_time_rate,%s,q=%d,Q=%s,success_rate=%s,n=%d,L=%d,#=%d.png', code_strategy, q, Q, alpha, block_length, list_size, max_num_indices_to_encode));
    end
end