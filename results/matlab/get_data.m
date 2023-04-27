function [A, max_qkd_leak] = get_data(code_strategy, q, Q, agg)
    p_err = str2double(Q);
    max_qkd_leak = 0.0;
    tol = 5 * eps(100); % A very small value
    if agg
        agg_str = ',agg';
    else
        agg_str = '';
    end
    A = readtable(sprintf("PycharmProjects/qsc_ir/results/%s,q=%d,p_err=%s%s.csv", code_strategy, q, Q, agg_str));
    switch q
        case 3
            switch Q
                case "0.0"
                    max_qkd_leak = log2(3);
                case "0.0001"
                    max_qkd_leak = 1.583713;
                case "0.001"
                    max_qkd_leak = 1.573464;
                case "0.01"
                    max_qkd_leak = 1.50306;
                case "0.02"
                    max_qkd_leak = 1.4411;
                case "0.05"
                    max_qkd_leak = 1.292;
            end
        case 5
            if strcmp(code_strategy, 'mb')
%                 A = A(A.N(:)>4000, :);
%                 A(isnan(A))=0;
%                 A = fillmissing(A,'constant',0,'DataVariables',@isinteger);
%                 A = fillmissing(A,'constant',0.0,'DataVariables',@isfloat);
%                 A = fillmissing(A,'constant','','DataVariables',@isstring);
%                 A = fillmissing(A,'constant',false,'DataVariables',@isboolean);
%                 A1 = readtable(sprintf("PycharmProjects/qsc_ir/results/%s,q=5%s.csv", code_strategy, agg_str));
%                 A1.is_fail = A1.is_success;
%                 A1.is_abort = A1.is_success;
%                 A1 = A1(A1.N(:)<4000, :);
%                 A = union(A, A1);
                A = readtable(sprintf("PycharmProjects/qsc_ir/results/%s,q=5%s.csv", code_strategy, agg_str));
            end
            switch Q
                case "0.0"
                    max_qkd_leak = log2(5);
                case "0.0001"
                    max_qkd_leak = 2.31987;
                case "0.001"
                    max_qkd_leak = 2.3025;
                case "0.01"
                    max_qkd_leak = 2.1678;
            end
        case 7
            A = readtable(sprintf("PycharmProjects/qsc_ir/results/%s,q=7%s.csv", code_strategy, agg_str));
            A = A(A.N(:)<1000, :);
            A2 = readtable(sprintf("PycharmProjects/qsc_ir/results/%s,q=7,p_err=%s%s.csv", code_strategy, Q, agg_str));
            A2 = A2(A2.N(:)>1000, :);
            A = union(A, A2);
    end
    A = A(ismembertol(A.q, q, tol) & ismembertol(A.p_err, p_err, tol) & A.N(:) < 8000, :);
end