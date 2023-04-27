function [B, G, group_names] = filter_and_group_data(code_strategy, A)
    switch code_strategy
        case 'mb'
            block_length=[3, 11, 19];
            list_size=[243,2187];
            max_num_indices_to_encode=[4, 8];
            tol = 5 * eps(100); % A very small value
            B = A(ismember(A.result_type, 'full_reduce') & ...
                ismembertol(A.mb_block_length(:), block_length, tol) & ...
                ismembertol(A.list_size(:), list_size, tol) & ...
                ismembertol(A.mb_max_num_indices_to_encode(:), max_num_indices_to_encode, tol), :);
            T = table(B.mb_block_length(:), B.list_size(:), B.mb_max_num_indices_to_encode);
            G = findgroups(T);
            group_names = string(B.mb_block_length) + ',' + string(B.list_size) + ',' + string(B.mb_max_num_indices_to_encode);
        case 'ldpc'
            sparsity = [2, 3, 4, 5];
            num_of_rounds=[30];
            tol = 0.05; % A very small value
            B = A(ismember(A.result_type, 'full_reduce') & ...
                ismembertol(A.ldpc_sparsity(:), sparsity, tol) & ...
                ismembertol(A.ldpc_max_num_rounds(:), num_of_rounds, tol), :);
            T = table(B.ldpc_sparsity(:), B.ldpc_max_num_rounds(:));
            G = findgroups(T);
            group_names = string(B.ldpc_sparsity) + ',' + string(B.ldpc_max_num_rounds);
        case 'polar'
            list_size = [243,2187];
            tol = 0.05; % A very small value
            % B = A(ismember(A.result_type, 'checked_list') & ...
            B = A(ismember(A.result_type, 'full_reduce') & ...
                ismembertol(A.list_size(:), list_size, tol), :);
            T = table(B.list_size(:));
            G = findgroups(T);
            group_names = string(B.list_size);
    end
end