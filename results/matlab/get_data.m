function A = get_data(code_strategy, q, p_err_str, agg, result_type, N, min_key_rate)
    if agg
        error("Agg currently not supported in thesis_plots");
    end

    A = get_data_helper(code_strategy, q, p_err_str, agg);

    % Basic filtering
    tol = 5 * eps(100); % A very small value

    A.N_2(:,1) = 2.^nextpow2(A.N);

    p_err = str2double(p_err_str);
    A = A(ismembertol(A.q, q, tol) & ...
        ismembertol(A.p_err, p_err, tol) & ...
        A.N(:) < 8000 & ...
        ismember(A.result_type, result_type), :);

    if ~isempty(N)
        A = A(ismembertol(A.N_2, N, tol), :);
    end

%     TODO: use elements of the following:
%     A = A(A.is_success == "True", :);
% When we want to filter by rate for ldpc and polar (some kind of relative_gap_rate)
%     A = A(ismember(A.N(:), N) & ...
%         ismembertol(A.key_rate(:), relative_gap_rate * theoretic_key_rate, 0.04), :);

    % Add new columns
    A.p_err_str(:,1) = p_err_str;

    A.max_qkd_leak(:,1) = 0.0;
    switch q
        case 3
            switch p_err_str
                case "0.0"
                    A.max_qkd_leak(:,1) = log2(3);
                case "0.0001"
                    A.max_qkd_leak(:,1) = 1.583713;
                case "0.001"
                    A.max_qkd_leak(:,1) = 1.573464;
                case "0.01"
                    A.max_qkd_leak(:,1) = 1.50306;
                case "0.02"
                    A.max_qkd_leak(:,1) = 1.4411;
                case "0.05"
                    A.max_qkd_leak(:,1) = 1.292;
            end
        case 5
            switch p_err_str
                case "0.0"
                    A.max_qkd_leak(:,1) = log2(5);
                case "0.0001"
                    A.max_qkd_leak(:,1) = 2.31987;
                case "0.001"
                    A.max_qkd_leak(:,1) = 2.3025;
                case "0.01"
                    A.max_qkd_leak(:,1) = 2.1678;
            end
    end

    switch code_strategy
        case 'mb'
            A.group_name(:,1) = string(A.mb_block_length) + ',' + string(A.list_size) + ',' + string(A.mb_max_num_indices_to_encode) + '-mb';
            A.group_number(:,1) = A.mb_block_length * max(A.mb_block_length) + A.list_size * max(A.list_size) + A.mb_max_num_indices_to_encode;
            A.code_strategy_number(:, 1) = 1;

            A.extra_cfg(:,1) = strings;
            success_rate_str_options = ["0.5", "0.75", "0.9", "0.99", "0.999", "0.9999", "1.0"];
            for success_rate_str = success_rate_str_options
                success_rate = str2double(success_rate_str);
                A.extra_cfg(ismembertol(A.mb_desired_success_rate, success_rate, tol), :) = success_rate_str;
            end
        case 'ldpc'
            A.group_name(:,1) = string(A.ldpc_sparsity) + '-ldpc,' +  string(A.ldpc_max_num_rounds) + '-bp';
            A.group_number(:,1) = A.ldpc_sparsity(:,1) * max(A.ldpc_sparsity) + A.ldpc_max_num_rounds(:,1);
            A.code_strategy_number(:, 1) = 2;
            A.extra_cfg(:,1) = string(A.ldpc_desired_relative_gap_rate);
        case 'polar'
            A.group_name(:,1) = string(A.list_size) + '-scl';
            A.group_number(:,1) = A.list_size;
            A.code_strategy_number(:, 1) = 3;
            A.extra_cfg(:,1) = string(A.polar_desired_relative_gap_rate);
    end

    switch code_strategy
        case 'mb'
            A.goal_key_rate(:,1) = 0.0;
        case 'ldpc'
            A.goal_key_rate(:,1) = A.ldpc_desired_relative_gap_rate .* A.theoretic_key_rate;
        case 'polar'
            A.goal_key_rate(:,1) = A.polar_desired_relative_gap_rate .* A.theoretic_key_rate;
    end
    if strcmp(code_strategy, 'polar') && ~isempty(min_key_rate)
        A = A(A.goal_key_rate >= min_key_rate, :);
    end

    % Add scaling exponent
    if agg
        A.gap(:,1) = A.theoretic_key_rate - A.key_rate_success_only;
        A.scalexp(:,1) = -log(A.N) ./ log(A.gap);
    else
        A.scalexp(:,1) = Inf;
        gap = A.theoretic_key_rate - A.key_rate;
        scalexp_rows = strcmp(A.is_success(:), 'True') & gap(:) > 0;
        A.scalexp(scalexp_rows, 1) = -log(A.N(scalexp_rows, 1)) ./ log(gap(scalexp_rows, 1));
    end

    % Fill empty columns to enable union
    mb_string_cols = {'mb_full_rank_encoding', ...
                      'mb_use_zeroes_in_encoding_matrix', ...
                      'mb_indices_to_encode_strategy', ...
                      'mb_rounding_strategy', ...
                      'mb_pruning_strategy', ...
                      'mb_radius_picking', ...
                      'mb_predetermined_number_of_encodings'};
    ldpc_string_cols = {'ldpc_decoder', ...
                        'ldpc_use_forking', ...
                        'ldpc_use_hints'};
    switch code_strategy
        case 'mb'
            A = convertvars(A, ldpc_string_cols, 'string');
        case 'ldpc'
            A = convertvars(A, mb_string_cols, 'string');
        case 'polar'
            A = convertvars(A, mb_string_cols, 'string');
            A = convertvars(A, ldpc_string_cols, 'string');
    end
end