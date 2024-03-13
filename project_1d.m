%% Exercise 3.1
close all
x = [1.0 2/3 0.0 1/3]; EToV = [3 4; 4 2; 2 1];
[x,EToV] = F.reordermesh1D(x,EToV);
P = 4;
[C] = F.ConstructConnectivity1D(EToV,P);
r = linspace(-1,1,P+1);
r = JacobiGL(0,0, P+1);
x1d = F.mesh1D(x,EToV,r);
C
plot(x1d,zeros(size(x1d)),'*')



%% Exercise 3.2
close all
u = @(x) exp(cos(x));
u = @(x) sin(x);
P = 2;


error_array = zeros(1,80);
for i_h = 2:4
H = i_h;
% Original mesh
x = linspace(0,3,H);
r = JacobiGL(0,0, P);
x1d = F.mesh1D(x,EToV,r);
ux = u(x1d);

% Mesh to interpolate on at each element from reference
NN = 20;
xx = linspace(-1,1,NN);
[~, V] = JacobiPP(r,0,0,P);
[~, VXX] = JacobiPP(xx,0,0,P);
INT = VXX*inv(V);
xN = F.mesh1D(x,EToV,xx); % Repeat these according to the outer grid

UX = reshape(ux, P+1,H-1);

u_int = INT*UX;

err = u_int(:) - u(xN);
error_array(i_h-1) = norm(err)/sqrt(length(xN));
end
%loglog(2:81,error_array); grid on; hold on
%loglog(1:81,0.00001./(1:81).^(P+2))

plot(xN,u(xN),'b','linewidth',2); hold on
plot(xN,u_int(:),'--r','linewidth',2)
% 
% figure(2);
% plot(xN, u(xN) - u_int(:))


%% Exercise 3.3

P = 1;
[r] = JacobiGL(0,0,P);
hn = 1.0;
[Kn, Mn] = F.stiffnessmatrix1D(hn,P,r)

%% Exercise 3.4
close all
L = 2;
c = 1;
d = exp(2);
u_true = @(x) exp(x);
M = 3;
P_array = 2:2:20;

error_array = zeros(size(P_array));
for i_P = 1:length(P_array)
    P = P_array(i_P);
    x = linspace(0,L,M+1);
    u_FEM = F.BVP1D(L,c,d,M);
    %[u_SEM,x1d, A,b,index1,index2] = F.BVP1Dhp(L,c,d,M,P);
    EToV = [1:M;2:M+1]';
    [u,x1d] = F.BVP1Dhp(L,c,d,M,P);
    residual = u-u_true(x1d); % The error is evaluated at the unif-GL node grid.
    error_array(i_P) = norm(residual)/sqrt(length(x1d));

end
figure(1)
semilogy(P_array, error_array)
title('p-convergence (spectral), h=3')

H_array = 2:5:100;
P = 3;
error_array = zeros(size(H_array));
for i_H = 1:length(H_array)
    H = H_array(i_H);
    x = linspace(0,L,M+1);
    u_FEM = F.BVP1D(L,c,d,M);
    %[u_SEM,x1d, A,b,index1,index2] = F.BVP1Dhp(L,c,d,M,P);
    EToV = [1:M;2:M+1]';
    [u,x1d] = F.BVP1Dhp(L,c,d,H,P);
    residual = u-u_true(x1d); % The error is evaluated at the unif-GL node grid.
    error_array(i_H) = norm(residual)/sqrt(length(x1d));

end
figure(1)
loglog(H_array, error_array); hold on
plot(H_array,H_array.^(-P-2))
title('h-convergence (1/H^{(P+2)}), P=3')





%% Exercise 3.5
close all
L = 3;
P = 60;
H = 2;
%f = @(x) exp(4*(x-1));
tau = 200;
%u_true = @(x) 1/16*(exp(4*(x-1)) - (x-1)*sinh(4)-cosh(4));
f = @(x) -sin(pi*x)*pi^2;
u_true = @(x) sin(pi*x)+1/2;
c = u_true(0);
d = u_true(L);

[u,x1d] = F.BVP1Dhp_pois_SBM_3_2(L,c,d,H,P,f,tau);


plot(x1d,u-u_true(x1d),'-b'); hold on


%% Test weak enforcement of BC
close all
L = 2;
P = 4;
tau = 100;
H = 10;
f = @(x) exp(4*(x-1));
u_true = @(x) 1/16*(exp(4*(x-1)) - (x-1)*sinh(4)-cosh(4));
c = u_true(0);
d = u_true(L);


[u,x1d] = F.BVP1Dhp_pois_SBM_3_2(L,c,d,H,P,f,tau);
plot(u-u_true(x1d))
%%


% H convergence
figure(1)
P = 3;
method = @(H) F.BVP1Dhp_pois_SBM_3_2(L,c,d,H,P,f,10*H);
H_array = 2:5:100;
error_array = F.convergence_test(method, H_array,u_true);
loglog(H_array,error_array)

% P convergence
figure(2)
H = 2;
method = @(P) F.BVP1Dhp_pois_SBM_3_2(L,c,d,H,P,f,tau);
P_array = 2:60;
error_array = F.convergence_test(method, P_array,u_true);
semilogy(P_array,error_array)

%plot(x1d,u,'-b'); hold on
%plot(x1d, u_true(x1d),'--r')

% Test convergence


%% Do SBM
close all
% I will consider the following true and surrogate domain: True domain is
% [-dd, L + dd]. The reference domain will be [0,L]. I do not think there
% is much use in the SBM in 1D tbh. But here we go...
L = 2;
P = 10;
H = 2;
tau = 10*H;
dd = +0.1;
f = @(x) -pi^2*sin(pi*x);
u_true = @(x) sin(pi*x)+ 1/2;
c = u_true(0-dd); % Notice that we now use these BCs...
d = u_true(L+dd); % Notice for d>0 the true domain i larger surrogate domain,so we do extrapolation

[u,x1d] = F.BVP1Dhp_pois_SBM_3_3(L,c,d,H,P,f,tau,dd);

% Next up: Set up an interpolation matrix... if we want to
% evaluate at a point xx, we need to be able to determine which element
% it belongs to.
x = linspace(0,L,H+1);
r = JacobiGL(0,0, P);
xx = linspace(-dd,L+dd,100);
INT = F.interp_matrix(xx,x,r);
u_xx = INT*u;
plot(xx, u_xx,'r'); hold on
plot(xx,u_true(xx),'--b')

% H convergence
P = 3;
r = JacobiGL(0,0, P);
H_array = 2:5:100;
error_array = zeros(size(H_array));
for i_H = 1:length(H_array)
    H = H_array(i_H);
    x = linspace(0,L,H+1);
    [u,x1d] = F.BVP1Dhp_pois_SBM_3_3(L,c,d,H,P,f,10*H,dd);
    [INT] = F.interp_matrix(xx, x, r);
    u_xx = INT*u;
    error_array(i_H) = norm(u_xx - u_true(xx(:)))/sqrt(length(xx));
end
figure(2)
loglog(H_array,error_array)

% P convergence
H = 3;
x = linspace(0,L,H+1);
P_array = 2:1:14;
error_array = zeros(size(P_array));
for i_P = 1:length(P_array)
    P = P_array(i_P);
    r = JacobiGL(0,0, P);
    [u,x1d] = F.BVP1Dhp_pois_SBM_3_3(L,c,d,H,P,f,10*H,dd);
    [INT] = F.interp_matrix(xx, x, r);
    u_xx = INT*u;
    error_array(i_P) = norm(u_xx - u_true(xx(:)))/sqrt(length(xx));
end
figure(3)
semilogy(P_array,error_array)

%%
H = 3;
P = 4;
x = linspace(0,L,H+1);
r = JacobiGL(0,0, P);
xx = linspace(0.01,L-0.01,100);
xx = xx(:);
[INT] = F.interp_matrix(xx, x, r)


%% H convergence
figure(2)
P = 3;
method = @(H) F.BVP1Dhp_pois_SBM_3_3(L,c,d,H,P,f,10*H,dd);
H_array = 2:5:100;
error_array = F.convergence_test(method, H_array,u_true);
loglog(H_array,error_array)

% P-convergence
figure(3)
H = 2;
method = @(P) F.BVP1Dhp_pois_SBM_3_3(L,c,d,H,P,f,10*H,dd);
P_array = 2:60;
error_array = F.convergence_test(method, P_array,u_true);
semilogy(P_array,error_array)
figure(1)

%%
close all;
H = 3;
P = 10;
L = 3;
u = @(x) exp(cos(x));
x = linspace(0,L,H+1);
r = JacobiGL(0,0, P);
x1d = F.mesh1D(x,EToV,r);
ux = u(x1d);
figure(1)
plot(x1d, ux); hold on

xxx = [0.4,1.2,1.8,2.1]';
find([diff([xxx<repmat(x,length(xxx),1)])]);


[element,~] = find(diff((xxx<repmat(x,length(xxx),1))'));

%%

dd = -0.5;
xx = [-dd, L+dd]; % scale this to the ref interval [-1,1]
xx_scaled = [-1-2*dd/(L/H),1+ 2*dd/(L/H)]
[~, V] = JacobiPP(r,0,0,P);
[~, VXX] = JacobiPP(xx_scaled,0,0,P);
INT = VXX*inv(V); % Add this interpolation matrix to the first row
A = zeros(size(INT)*(H));
A(1,1:P+1) = INT(1,:);
A(end,end-(P+1)+1:end) = INT(2,:);
u_int = A*ux;
plot(xx, u_int([1,end]),'o')
plot(linspace(0,L,100),u(linspace(0,L,100)))

%% Test the interp_matrix function
close all
L = 5;
H = 5;
P = 9;
u = @(x) exp(cos(x));
r = JacobiGL(0,0,P);
x = linspace(0,L,H+1);
x1d = F.mesh1D(x,EToV,r);
x1d(P+1:P+1:(H-1)*(P+1))=[];
ux = u(x1d);
plot(x1d,ux); hold on


xx = linspace(-1,L+1,100);
INT = F.interp_matrix(xx, x, r);

u_interp = INT*ux;
plot(xx,u_interp)
plot(xx,u(xx))

%%
            Vr = F.GradJacobiP(r,0,0,P);
            [~, V] = JacobiPP(r,0,0,P);
            Dr = Vr*inv(V);
            Dn = Dr*2/hn;
            M = inv(V*V.');
            Kn = 2/hn*Dr.'*M*Dr;
            Mn = hn/2*M;
            size(Dr)
            size(M)
            size(Dr.'*M*Dr)

%%
[Kn,Mn,Dn] = F.stiffnessmatrix1D(hn,P,r);
rank(Kn)
size(Kn)

