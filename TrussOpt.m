clear all;close all;clc;
global data;
displayIts = true;
% From skeleton Code %
data.storedesigns=[];


% Provided Values
data.L = 1000; % mm
data.As = 1e3; % mm^2
data.E = 2.5e5; % MPa
data.F = 1e4; % N
% Decided to store cross-sectional areas of the elements in this format,
% in a similar fashion to a 1st order tensor
data.A = data.As * [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]';

% More or Less Unnecessary as long as the required values are provided
data.Mat = {data.L, data.As, data.E}; % Mat: cell array

% Connectivity Matrix
data.CEM = [1,2;2,3;3,4;4,5;5,6;6,7;7,10;8,10;
    8,9;1,9;1,8;2,8;3,8;7,8;3,7;4,7;5,7];
% Constrained Nodes
data.Constr = [2,9,10];
% Position Vector of the Nodes divided by the length of the struts
data.X = [0,0;2,0;3,0;4,0;6,0;6,2;4,2;2,2;0,2;3,3.5];
% Position Vector of the Nodes times the length of the struts
data.r = data.L * data.X;
% Applied External Force Vector 
data.Fext = zeros(size(data.CEM,2)*size(data.X, 1), 1);
data.Fext(19) = -data.F;

% Length of a given element
[l] = EleLength(data.CEM, data.r);

% Initial Volume
data.V0 = data.A' * l;

 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fmincon - optimize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initial Design
x0 =  data.A;  % Squaresections A_i

% objective  und constraints
fhf = @(x) objective(x,data);
fhgtb = @(x) constraints(x,data);

% lower and upper bound - Gleichheitsnebenbedingungen 
lb = ones(size(x0)) * .1;
ub = ones(size(x0)) * 2000;

% fmincon Options 
if displayIts
    options = optimoptions('fmincon','Display','iter',...
        'SpecifyObjectiveGradient', true,... % switch T/F to check sol
        'SpecifyConstraintGradient', true, ...
        'Algorithm', 'interior-point',...
        'OutputFcn', @outfun); 
else
    options = optimoptions('fmincon',...
    'SpecifyObjectiveGradient', true,...
    'SpecifyConstraintGradient', true, ...
    'Algorithm', 'interior-point',...
    'OutputFcn', @outfun); 
end

% Optimiere x
x_sol = fmincon(fhf,x0,[],[],[],[],lb,ub,fhgtb,options);

%% Visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup
num_elements = size(data.CEM, 1); % for convinence

% Title and Labels for Cross Sectional Areas Plot
figure;
ax1 = subplot(2, 1, 1);
hold on;
axis equal;
xlabel('x (mm)');
xlim([min(data.r(:,1)) max(data.r(:,1))]);
ylabel('y (mm)');
ylim([min(data.r(:,2)) max(data.r(:,2))]);
yticks(min(data.r(:,2)):1e3/2:max(data.r(:,2)))
title('Cross-Sectional Areas of Struts')
% Define the colormap
cmap1 = colormap(ax1, flipud(gray));
% Create the colorbar
cb1 = colorbar(ax1);
cb1 = flipud(cb1);
cb1.Label.String = 'A in [mm^{2}]';
% Set the ticks and labels on the colorbar
% x_sort = unique(x_sol,'sorted');
% cb1.Ticks = x_sort;
% Mappings of the stress for the struts
%n = size(gray,1);

%%% Labels for Stress Plot %%%%
[u] = FEMTruss(data.r, data.CEM, data.Mat, x_sol, data.Fext, data.Constr);
[Sigma] = Stress(u, data.CEM, data.r, x_sol, data.Mat);
% Title and Labels for Stress Plot
ax2 = subplot(2, 1, 2);
hold on; axis equal;
xlabel('x (mm)');
xlim([min(data.r(:,1)) max(data.r(:,1))]);
ylabel('y (mm)');
ylim([min(data.r(:,2)) max(data.r(:,2))]);
yticks(min(data.r(:,2)):1e3/2:max(data.r(:,2)))
title('Stresses in Struts')
% Define the colormap
cmap2 = colormap(ax2, jet);
% Create the colorbar and label
cb2 = colorbar(ax2);
cb2.Label.String = '\sigma in [MPa]';
% Set the ticks and tick labels on the colorbar
% sigma_sort = unique(Sigma,'sorted');
% cb2.Ticks = sigma_sort;
% Mappings of the stress for the struts
n = size(jet,1);

Atensor = data.storedesigns;

for i=1:size(Atensor,2)
    A_i = Atensor(:,i);
    if max(A_i) == min(A_i)
       A_map = A_i / A_i * (n-1) + 1;
       clim(ax1, [0 max(A_i)]);
    else
A_map = (A_i - min(A_i))/(max(A_i)- min(A_i)) * (n-1) + 1;
clim(ax1, [min(A_i) max(A_i)]);
    end

[u_i] = FEMTruss(data.r, data.CEM, data.Mat, A_i, data.Fext, data.Constr);
[Sigma_i] = Stress(u_i, data.CEM, data.r, A_i, data.Mat);


clim(ax2, [min(Sigma_i) max(Sigma_i)]);

sigma_i_map = (Sigma_i - min(Sigma_i))/(max(Sigma_i)-min(Sigma_i)) * (n-1) + 1;

cb1.Ticks = [];
A_sort = unique(A_i,'sorted');
cb1.Ticks = A_sort;


cb2.Ticks = [];
sigma_sort = unique(Sigma_i,'sorted');
cb2.Ticks = sigma_sort;

% Plotting
for j=1:num_elements
    [~,x_ele,~] = ElementMap(data.CEM, data.r, j);
    % Plot with color corresponding to stress
    plot(ax1, x_ele(:, 1), x_ele(:, 2), 'LineWidth', 2, 'Color', ...
         cmap1(floor(A_map(j)), :),'Marker','o');
    pause(1e-6);

    % Plot with color corresponding to stress
    plot(ax2, x_ele(:, 1), x_ele(:, 2), 'LineWidth', 2, 'Color', ...
         cmap2(floor(sigma_i_map(j)), :),'Marker','o');
    pause(1e-6)
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Compliance Function and its Gradient %%%%%%%%%%%%%%%%%%%
function [f,df] = objective(x,data)

% Does FE Analysis, similar to FEASolverTruss Function but now, for input 
% variable A the design vector: x
% x = Anew, where Anew is the updated cross-sections
[u] = FEMTruss(data.r, data.CEM, data.Mat, x, data.Fext, data.Constr);
% Lengths of elements
[l] = EleLength(data.CEM, data.r);
% compliance functions f (or c)
f = 0;
% Derivative of the compliance function, df/dA (or dc/dA), Anew = x
df = zeros(size(l,1),1);
for i=1:size(data.A, 1) % iterates through numbers of elements
   [~, x_ele, idx_dof_ele] = ElementMap(data.CEM, data.r, i);
   te = LocalTM(x_ele); % Local Transformation Matrix te
   ke = data.E*x(i)/l(i) * [1 -1;-1 1]; % Local Stiffness Matrix ke
   ue = te'*u(idx_dof_ele); % Local displacements ue
% Computes compliance function and its gradient 
   f = f + ue' * ke * ue; 
   df(i,1) = -ue' * data.E/l(i) * [1 -1;-1 1] * ue;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%% Constraint %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [g,geq,dgdx,dgeqdx] = constraints(x,data)
% Input:
%       x       ......      design variables (Ai)
%       data    ......      struct holding all data needed
%
% Output:
%       g       ......      inequality constraints (could be empty)
%       geq     ......      equatilty constraints

% provide gradients 

% Element Lengths
[l] = EleLength(data.CEM, data.r);

% This is the constraint that is subjected to the system g(x), 
% where x = Anew
% Below is the derivative of g with respect to the design vector x
    geq = x' * l - data.V0;
   dgeqdx = l;
   
    g=[];
    dgdx = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%% FEM Truss Solver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u] = FEMTruss(x, CEM, Mat, A, Fa, Constr)
% A is now a first order tensor instead of apply Mat values
dim = size(x, 2); % dimesion is assigned
num_nodes = size(x,1); % number of nodes
num_elements = size(CEM,1); % number of elements
% Modulus of Elascity
E = Mat{3};

% Global K Matrix
K = zeros(dim*num_nodes, dim*num_nodes);

for i=1:num_elements
% Organization of elements %
   [~, x_ele, idx_dof_ele] = ElementMap(CEM, x, i);

    % Material Properties and Characteristics of an element
    E_ele = E;
    A_ele = A(i);

   Ke = LocalSM(x_ele, E_ele, A_ele); % N/mm
   Te = LocalTM(x_ele);

% Assembling of the Global K (stiffness) Matrix
    K(idx_dof_ele, idx_dof_ele) = ...
        K(idx_dof_ele, idx_dof_ele) + Te * Ke * Te'; % N/mm
end

% Creates an array of unconstrained nodes
free_dof = true(dim*num_nodes, 1);
free_dof(Constr) = false;

% Global F vector
F = zeros(dim*num_nodes, 1); % Global Force Vector
F = F + Fa; % N
% Reducing Matrices and Vectors
ReduceK = K(free_dof, free_dof);
ReduceF = F(free_dof);
Reduceu = ReduceK\ReduceF;

% Solutions %
u = zeros(dim*num_nodes,1);
u(free_dof) = Reduceu; % mm

end


%%%%%%%%%%%%%%%%%%%%%%%%%% Stress Calcuation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Sigma] = Stress(u, CEM, x, A, Mat)
% Calculating Stresses
num_elements = size(CEM, 1);
E = Mat{3};
Sigma = zeros(num_elements,1);
for i=1:num_elements
    [~, x_ele, idx_dof_ele] = ElementMap(CEM, x, i);
    % E and A for a given element
    E_ele = E;
    A_ele = A(i);
% Local Stiffness Matrix
   ke = LocalSM(x_ele, E_ele, A_ele);
% Local Transformation Matrix
    te = LocalTM(x_ele);
    % local displacements 
    ue = te' * u(idx_dof_ele);
    % local forces
    fe = ke * ue;
    % Local Stresses
    sigmae = fe(1)/A_ele;
    % Stresses stored as a column vector
    Sigma(i) = sigmae;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%% 2D Element Mapping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [CEM_ele, x_ele, idx_dof_ele] = ElementMap(CEM, x, i)
    dim = size(x,2); % Dimension of system
% Mappings of nodes to elements and degree of freedoms to elements
    CEM_ele = CEM(i,:);
    % nodes positions
    x_ele = x(CEM_ele,:);
    % degrees of freedom for an element
    idx_dof_ele = [dim * CEM_ele(1)-1, dim * CEM_ele(1),...
        dim * CEM_ele(2)-1, dim * CEM_ele(2)];
end
%%%%%%%%%%%%%%%%%%%%%%% Local Stiffness Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ke] = LocalSM(x_ele, E_ele, A_ele)
% Ke = EA/L * [1, -1; -1, 1]
% Stiffness matrix without material properties (B matrix)
b = [1,-1;-1,1];
   
    % These two have to be calculated, but are from known values/connectity
    % matrix
    L_ele = norm(x_ele(2,:)'-x_ele(1,:)');
    % Local Stiffness Matrix
    Ke = E_ele * A_ele/L_ele * b;
end
%%%%%%%%%%%%%%%%%%%%% Local Transformation Matrix %%%%%%%%%%%%%%%%%%%%%%%%%
function te = LocalTM(x_ele)

% Local transformation matrix
phi_ele = atan((x_ele(2,2)-x_ele(1,2))/(x_ele(2,1)-x_ele(1,1)));
te = [cos(phi_ele) 0 ;
      sin(phi_ele) 0 ;
      0 cos(phi_ele) ;
      0 sin(phi_ele)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Length of Elements %%%%%%%%%%%%%%%%%%%%%%%%%%
function [l] = EleLength(CEM, x)
l = zeros(size(CEM,1),1);
for i=1:size(CEM,1)
    [~, x_ele, ~] = ElementMap(CEM, x, i);
    l(i) = norm(x_ele(2,:)'-x_ele(1,:)');
end
end
%%%%%%%%%%%%%%% Store Design Values throught Iterations %%%%%%%%%%%%%%%%%%%
function stop = outfun(x,optimValues,state)
global data;
%       stop = outfun(x,optimValues,state)
%
%       Function - Handel zum loggen der FMINCON States
%
%   INPUT: 
%
%       x           .....   aktuelles Design
%       optimValues .....   aktuelle Vars
%       state       .....   FMINCON State
%
%   OUTPUT:
%
%       stop        .....   Flag
%
stop = false;
data.storedesigns=[data.storedesigns,x];

% stop = false;
% 
%    switch state
%        case 'iter'
%            data.log.iter = [data.log.iter; optimValues.iteration];
%            data.log.x = [data.log.x, x];
%            data.log.gradient = [data.log.gradient, optimValues.gradient];
%            data.log.fval = [data.log.fval; optimValues.fval];     
%        otherwise
%    end
% end
end