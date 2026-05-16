function [O1, O2] = PMX(P1, P2)
% PMX - Partially Mapped Crossover
%   [O1, O2] = PMX(P1, P2)
%   O1 = PMX(P1, P2)

    [D, n] = size(P1);

    if ~isequal(size(P1), size(P2))
        error('P1 and P2 must have the same dimensions (D x n)');
    end

    O1 = zeros(D, n);
    need_o2 = (nargout > 1);
    if need_o2
        O2 = zeros(D, n);
    end

    mapping = zeros(D, 1);

    if need_o2
        % --- Full PMX: both offspring in one pass ---
        for i = 1:n
            p1 = P1(:, i);
            p2 = P2(:, i);

            c1 = randi(D);
            c2 = randi(D);
            left  = min(c1, c2);
            right = max(c1, c2);

            % Offspring 1
            o1 = p1;
            o1(left:right) = p2(left:right);
            mapping(:) = 0;
            mapping(p2(left:right)) = p1(left:right);
            for j = 1:left-1
                val = p1(j);
                while mapping(val) > 0, val = mapping(val); end
                o1(j) = val;
            end
            for j = right+1:D
                val = p1(j);
                while mapping(val) > 0, val = mapping(val); end
                o1(j) = val;
            end
            O1(:, i) = o1;

            % Offspring 2
            o2 = p2;
            o2(left:right) = p1(left:right);
            mapping(:) = 0;
            mapping(p1(left:right)) = p2(left:right);
            for j = 1:left-1
                val = p2(j);
                while mapping(val) > 0, val = mapping(val); end
                o2(j) = val;
            end
            for j = right+1:D
                val = p2(j);
                while mapping(val) > 0, val = mapping(val); end
                o2(j) = val;
            end
            O2(:, i) = o2;
        end
    else
        % --- O1 only: lean loop ---
        for i = 1:n
            p1 = P1(:, i);
            p2 = P2(:, i);

            c1 = randi(D);
            c2 = randi(D);
            left  = min(c1, c2);
            right = max(c1, c2);

            o1 = p1;
            o1(left:right) = p2(left:right);
            mapping(:) = 0;
            mapping(p2(left:right)) = p1(left:right);
            for j = 1:left-1
                val = p1(j);
                while mapping(val) > 0, val = mapping(val); end
                o1(j) = val;
            end
            for j = right+1:D
                val = p1(j);
                while mapping(val) > 0, val = mapping(val); end
                o1(j) = val;
            end
            O1(:, i) = o1;
        end
    end
end