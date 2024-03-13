classdef F
    methods(Static)

        function [x,EToVnew] = reordermesh1D(x,EToV)
            [x, index] = sort(x);
            EToVnew = zeros(size(EToV));
            for i=1:length(index)
                EToVnew(EToV == index(i)) = i;
            end
        end

        function [C] = ConstructConnectivity1D(EToV,P)
            N = size(EToV);
            N = N(1);
            Mp = P+1; % Number of nodes per element
            C = zeros(N,Mp);
            gidx = 2;
            for n=1:N
                gidx = gidx - 1;
                for i=1:Mp
                    C(n,i) = gidx;
                    gidx = gidx + 1;
                end
            end
        end

        function [x1d] = mesh1D(x,EToV,r)
            x1d = [];
            for i_x = 1:length(x)-1
                b = x(i_x+1);
                a = x(i_x);
                rx = (r + 1)*(b-a)/2+a;
                rx = rx(:);
                x1d = [x1d; rx];
            end
        end

        function [Vx] = GradJacobiP(x,a,b,N)
            if N==1
                Vp1 = JacobiPP(x,a+1,b+1,N-1);
            else
                [~,Vp1] = JacobiPP(x,a+1,b+1,N-1);
            end
            Vp1 = [zeros(size(Vp1(:,1))), Vp1];
            nn = 0:(N);
            coef = sqrt(nn.*(nn+a+b+1));
            Vx = repmat(coef,length(x),1).*Vp1;
        end

        function [Kn,Mn,Dn] = stiffnessmatrix1D(hn,P,r)
            Vr = F.GradJacobiP(r,0,0,P);
            [~, V] = JacobiPP(r,0,0,P);
            Dr = Vr*inv(V);
            Dn = Dr*2/hn;
            M = inv(V*V.');
            Kn = 2/hn*Dr.'*M*Dr;
            Mn = hn/2*M;
        end

        function [u,x1d] = BVP1Dhp(L,c,d,M,P) % For the problem u''-u=0, exercise 3.4
            M_P = P+1;
            EToV = [1:M;2:M+1]';
            x = linspace(0,L,M+1);
            hn = x(2)-x(1);
            r = JacobiGL(0,0,P);
            x1d = F.mesh1D(x,EToV,r);
            x1d(P+1:P+1:(M-1)*(P+1))=1000;
            x1d = x1d(x1d~=1000);

            [C] = F.ConstructConnectivity1D(EToV,P);
            N = max(max(C));
            [Kn,Mn] = F.stiffnessmatrix1D(hn,P,r); % Compute outside for loop as they are constant over elements
            
            N_nonzero = M*(M_P+1)*M_P/2;
            index1 = zeros(N_nonzero,1);
            index2 = zeros(N_nonzero,1);
            A_value = zeros(N_nonzero,1);
            
            kn = +Kn + Mn; % <----- LHS of PDE.
            count = 0;
            for n=1:M
                for j =1:M_P
                    for i=1:j
                        count = count + 1;
                        index1(count) = C(n,i);
                        index2(count) = C(n,j);
                        A_value(count) = kn(i,j);
                    end
                end

            end
            A = sparse(index1, index2, A_value, N,N);

            % The indexing is slow here, but i beleive that's ok, since we
            % only modify a small part of the array
            b = zeros(N,1);
            b(1) = c;
            A(1,1) = 1;
            for i=2:M_P
                b(i) = b(i) - A(1,i)*c;
                A(1,i) = 0;
            end
            b(C(end,i)) = d;
            for i=1:M_P-1
                b(C(end,i)) = b(C(end,i)) - A(C(end,i), C(end,M_P))*d;
                A(C(end,i), C(end,end)) = 0;
            end
            A(C(end,end), C(end,end)) = 1;
            b(1) = c;
                    [U,flag] = chol(A);
            if flag == 0
            u = U \ (U' \ b);
            else
            disp('A is not positive definite'), return
            end
            % OUTPUT
        end

        function [u, x1d] = BVP1Dhp_pois_FEM_3_5(L,c,d,M,P,f) % For u'' = f, exercise 3.5
            % F Should be a function handle for the RHS
            M_P = P+1;
            EToV = [1:M;2:M+1]';
            x = linspace(0,L,M+1);
            hn = x(2)-x(1);
            r = JacobiGL(0,0,P);
            x1d = F.mesh1D(x,EToV,r);
            x1d(P+1:P+1:(M-1)*(P+1))=[];

            
            [C] = F.ConstructConnectivity1D(EToV,P);
            N = max(max(C));
            [Kn,Mn] = F.stiffnessmatrix1D(hn,P,r); % Compute outside for loop as they are constant over elements
            
            N_nonzero = M*M_P^2;
            index1 = zeros(N_nonzero,1);
            index2 = zeros(N_nonzero,1);
            A_value = zeros(N_nonzero,1);
            M_value = zeros(N_nonzero, 1);
            
            kn = +Kn; % <----- LHS of PDE.
            mn = +Mn;
            count = 0;
            for n=1:M
                for j =1:M_P
                    for i=1:M_P
                        count = count + 1;
                        index1(count) = C(n,i);
                        index2(count) = C(n,j);
                        A_value(count) = kn(i,j);
                        M_value(count) = mn(i,j);
                    end
                end
            end
            A = sparse(index1, index2, A_value, N,N);
            M = sparse(index1, index2, M_value, N,N);

            % The indexing is slow here, but i beleive that's ok, since we
            % only modify a small part of the array
            %b = zeros(N,1);
            b = -M*f(x1d(:));
            b(1) = c;
            A(1,1) = 1;
            for i=2:M_P
                b(i) = b(i) - A(1,i)*c;
                A(1,i) = 0;
            end
            b(C(end,i)) = d;
            for i=1:M_P-1
                b(C(end,i)) = b(C(end,i)) - A(C(end,i), C(end,M_P))*d;
                A(C(end,i), C(end,end)) = 0;
            end
            A(C(end,end), C(end,end)) = 1;
            b(1) = c;
                    [U,flag] = chol(A);
            if flag == 0
            u = U \ (U' \ b);
            else
            disp('A is not positive definite'), return
            end
            % OUTPUT
        end

        function [u, x1d] = BVP1Dhp_pois_SBM_3_1(L,c,d,M,P,f) % For u'' = f, exercise 3.1 from SBM
            % F Should be a function handle for the RHS
            M_P = P+1;
            EToV = [1:M;2:M+1]';
            x = linspace(0,L,M+1);
            hn = x(2)-x(1);
            r = JacobiGL(0,0,P);
            x1d = F.mesh1D(x,EToV,r);
            x1d(P+1:P+1:(M-1)*(P+1))=[];

            
            [C] = F.ConstructConnectivity1D(EToV,P);
            N = max(max(C));
            [Kn,Mn] = F.stiffnessmatrix1D(hn,P,r); % Compute outside for loop as they are constant over elements
            
            N_nonzero = M*M_P^2;
            index1 = zeros(N_nonzero,1);
            index2 = zeros(N_nonzero,1);
            A_value = zeros(N_nonzero,1);
            M_value = zeros(N_nonzero, 1);
            
            kn = +Kn; % <----- LHS of PDE.
            mn = +Mn;
            count = 0;
            for n=1:M
                for j =1:M_P
                    for i=1:M_P
                        count = count + 1;
                        index1(count) = C(n,i);
                        index2(count) = C(n,j);
                        A_value(count) = kn(i,j);
                        M_value(count) = mn(i,j);
                    end
                end
            end
            A = sparse(index1, index2, A_value, N,N);
            M = sparse(index1, index2, M_value, N,N);

            % The indexing is slow here, but i beleive that's ok, since we
            % only modify a small part of the array
            %b = zeros(N,1);
            b = -M*f(x1d(:));
            b(1) = c;
            A(1,1) = 1;
            for i=2:M_P
                b(i) = b(i) - A(1,i)*c;
                A(1,i) = 0;
            end
            b(C(end,i)) = d;
            for i=1:M_P-1
                b(C(end,i)) = b(C(end,i)) - A(C(end,i), C(end,M_P))*d;
                A(C(end,i), C(end,end)) = 0;
            end
            A(C(end,end), C(end,end)) = 1;
            b(1) = c;
                    [U,flag] = chol(A);
            if flag == 0
            u = U \ (U' \ b);
            else
            disp('A is not positive definite'), return
            end
            % OUTPUT
        end

        function [u, x1d] = BVP1Dhp_pois_SBM_3_2(L,c,d,M,P,f,tau) % For u'' = f, weak enforcement
            % F Should be a function handle for the RHS
            M_P = P+1;
            EToV = [1:M;2:M+1]';
            x = linspace(0,L,M+1);
            hn = x(2)-x(1);
            r = JacobiGL(0,0,P);
            x1d = F.mesh1D(x,EToV,r);
            x1d(P+1:P+1:(M-1)*(P+1))=[];

            [C] = F.ConstructConnectivity1D(EToV,P);
            N = max(max(C));
            [Kn,Mn, Dn] = F.stiffnessmatrix1D(hn,P,r); % Compute outside for loop as they are constant over elements
            
            N_nonzero = M*M_P^2;
            index1 = zeros(N_nonzero,1);
            index2 = zeros(N_nonzero,1);
            A_value = zeros(N_nonzero,1);
            M_value = zeros(N_nonzero, 1);
            D_value = zeros(N_nonzero, 1);

            
            kn = +Kn; % <----- LHS of PDE.
            mn = +Mn;
            dn = Dn;
            count = 0;
            for n=1:M
                for j =1:M_P
                    for i=1:M_P
                        count = count + 1;
                        index1(count) = C(n,i);
                        index2(count) = C(n,j);
                        A_value(count) = kn(i,j);
                        M_value(count) = mn(i,j);
                        D_value(count) = dn(i,j);
                    end
                end
            end
            A = sparse(index1, index2, A_value, N,N);
            M = sparse(index1, index2, M_value, N,N);
            D = sparse(index1, index2, D_value, N,N);

            % The indexing is slow here, but i beleive that's ok, since we
            % only modify a small part of the array
            %b = zeros(N,1);
            b = -M*f(x1d(:));

            [~, V] = JacobiPP(r,0,0,P);
            Vx = F.GradJacobiP(r,0,0,P);

            A(1,1) = A(1,1)+tau; % Add tau to upper left corner
            A(C(end,end), C(end,end)) = A(C(end,end), C(end,end)) + tau; % Add tau to lower left corner

            for i=1:M_P
                A(C(1,1), C(1,i)) = A(C(1,1), C(1,i)) + D(C(1,1), C(1,i));    % Add D to first row
            end

            for i=M_P:-1:1
                A(C(end,end), C(end,i)) = A(C(end,end), C(end,i)) - D(C(end,end), C(end,i));     % Add D to last row
            end

            b(1) = b(1) + tau*c; b(end) = b(end) + tau*d;

            u = A\b;
            % OUTPUT
        end
        
        function [u, x1d] = BVP1Dhp_pois_SBM_3_3(L,c,d,M,P,f,tau,dd) % For u'' = f, weak enforcement
            % F Should be a function handle for the RHS
            M_P = P+1;
            EToV = [1:M;2:M+1]';
            x = linspace(0,L,M+1);
            hn = x(2)-x(1);
            r = JacobiGL(0,0,P);
            x1d = F.mesh1D(x,EToV,r);
            x1d(P+1:P+1:(M-1)*(P+1))=[];
            
            [C] = F.ConstructConnectivity1D(EToV,P);
            N = max(max(C));
            [Kn,Mn, Dn] = F.stiffnessmatrix1D(hn,P,r); % Compute outside for loop as they are constant over elements
            
            N_nonzero = M*M_P^2;
            index1 = zeros(N_nonzero,1);
            index2 = zeros(N_nonzero,1);
            A_value = zeros(N_nonzero,1);
            M_value = zeros(N_nonzero, 1);
            D_value = zeros(N_nonzero, 1);

            
            kn = +Kn; % <----- LHS of PDE.
            mn = +Mn;
            dn = Dn;
            count = 0;
            for n=1:M
                for j =1:M_P
                    for i=1:M_P
                        count = count + 1;
                        index1(count) = C(n,i);
                        index2(count) = C(n,j);
                        A_value(count) = kn(i,j);
                        M_value(count) = mn(i,j);
                        D_value(count) = dn(i,j);
                    end
                end
            end
            A = sparse(index1, index2, A_value, N,N);
            MM = sparse(index1, index2, M_value, N,N);
            D = sparse(index1, index2, D_value, N,N);

            % The indexing is slow here, but i beleive that's ok, since we
            % only modify a small part of the array
            %b = zeros(N,1);
            b = -MM*f(x1d(:));

            xx = [-dd, L+dd]; % scale this to the ref interval [-1,1]
            xx_scaled = [-1-2*dd/(L/M),1+ 2*dd/(L/M)];
            [~, V] = JacobiPP(r,0,0,P);
            [~, VXX] = JacobiPP(xx_scaled,0,0,P);
            INT = VXX*inv(V); % Add this interpolation matrix to the first row
            % and last row... (in the corners of the rows

            for i=1:M_P
                A(C(1,1), C(1,i)) = A(C(1,1), C(1,i)) + D(C(1,1), C(1,i));    % Add D to first row
                A(C(1,1), C(1,i)) = A(C(1,1), C(1,i)) + tau*INT(1,i); 
            end

            for i=M_P:-1:1
                A(C(end,end), C(end,i)) = A(C(end,end), C(end,i)) - D(C(end,end), C(end,i));     % Add D to last row
                A(C(end,end),C(end,i)) = A(C(end,end),C(end,i)) + tau*INT(end,i);

            end
        
            b(1) = b(1) + tau*c; b(end) = b(end) + tau*d;
            

            u = A\b;
            % Correct the output mesh
            %x = linspace(-dd,L+dd,M+1)-1;
            %x1d = F.mesh1D(x,EToV,r);
            %x1d(P+1:P+1:(M-1)*(P+1))=[];
            % OUTPUT
        end
        %
        function [error_array] = convergence_test(method, parameter_array,u_true)
            N = length(parameter_array);
            error_array = zeros(1,N);
            for i_N=1:N
                [u,x1d] = method(parameter_array(i_N));
                residual = u-u_true(x1d); % The error is evaluated at the unif-GL node grid.
                error_array(i_N) = norm(residual)/sqrt(length(x1d));
            end
        end


        % The old FEM function from the FEM course
        function [u] = BVP1D(L,c,d,M)
        % Purpose: Solve second-order boundary value problem using FEM.
        % Author(s): Martin Carøe
        % INPUT PARAMETERS
        % L : Domain length
        % c : Left boundary condition
        % d : Right boundary condition
        % x : 1D mesh vector x(1:{M}), or integer x=M for the number of mesh points
        % plot_FEMsolution : Logical. True if the solution is to be plotted
        % GLOBAL ASSEMBLY
        % Assemble A (the upper triangle only) and b. (Algorithm 1)
        x = linspace(0,L,M+1);
        M = length(x);
        h = diff(x);
        A = spalloc(M,M,2*M-1);
        % Very slow implementation
        for i=1:(M-1)
            hi = h(i);
            k11 = 1/hi + hi/3; 
            k12 = -1/hi + hi/6;
            A(i,i) = A(i,i) + k11;
            A(i,i+1) = k12;
            A(i+1,i+1) = k11;
        end
        % IMPOSE BOUNDARY CONDITIONS
        % (Algorithm 2)
        b = zeros(M,1);
        b(1) = c; b(2) = b(2)-A(1,2)*c;
        A(1,1) = 1; A(1,2) = 0;
        b(M) = d; b(M-1) = b(M-1) - A(M-1,M)*d;
        A(M,M) = 1; A(M-1,M) = 0;
        % SOLVE SYSTEM
        % Solve using the Cholesky factorization of A to solve A*u=b
        [U,flag] = chol(A);
        if flag == 0
        u = U \ (U' \ b);
        else
        disp('A is not positive definite'), return
        end
        % OUTPUT
        end


        function [INT] = interp_matrix(xx, x, r)
            % xx: Evaluate function on xx.
            % x: Outer mesh defining the elements
            % EToV:...
            P = length(r)-1;
            M_P = P+1;
            N = length(xx);
            M = length(x)-1;
            xx = xx(:);
            xx_temp = xx(xx>x(1) & xx<x(end));
            element = zeros(size(xx));
            [element_temp,~] = find(diff((xx<repmat(x,length(xx),1))'));
            element(xx>x(1) & xx<x(end)) = element_temp;
            element(xx<=x(1)) = 1;
            element(xx>=x(end)) = M;
            EToV = [1:M;2:M+1]';
            C = F.ConstructConnectivity1D(EToV,P);

            N_nonzero = N*M_P;
            index1 = zeros(N_nonzero,1);
            index2 = zeros(N_nonzero,1);
            int_value = zeros(N_nonzero,1);
            
            [~, V] = JacobiPP(r,0,0,P);
            [~, VXX] = JacobiPP(xx,0,0,P);
            ab = x(EToV);
            count = 0;
            xx_scaled = zeros(size(xx));
            
            for i=1:N
                a = ab(element(i),1);
                b = ab(element(i),2);
                xx_scaled(i) = 2*(xx(i)-a)/(b-a) -1;
            end
            [~, V] = JacobiPP(r,0,0,P);
            [~, VXX] = JacobiPP(xx_scaled,0,0,P);
            int = VXX*inv(V); % Size is length(xx), M_P

            for i=1:N
                for j=1:M_P
                    count = count+1;
                    index1(count) = i;
                    index2(count) = C(element(i),j);
                    int_value(count) = int(i,j);
                end
            end
            INT = sparse(index1, index2, int_value, N,max(max(C)));

        end

    end
end

