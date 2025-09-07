%% Truss Structure FEA Solver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Structure (system) is comprimsed of 1D beams/rods/struts
% This system's coordinates are in the Global (Design) Domain.
% The struts can have varying cross-sectionals areas (A) as well as 
% varying Modulus' of Elasticty (E), and are in 2D (2 degrees of freedom at 
% each node i.e. ux and uy; Fx and Fy
%
%% Inputs: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1). Initial Positions of the Nodes (X) for the struts in this fashion:
% X = [x1, y1; ...; xn, yn] where it is untiless.
% Note: x is the initial Positions of the Nodes with units of L such that:
% x = L * X
% 
% 2). Connectivity Matrix (CEM) is setup as as such
% CEM = [ "1st strut nodes", ... , "nth strut nodes"]'
% The number assigned to a node is not relavanet as along as the indexs of 
% the position vector (X) is consist with the formatting of the 
% Connecitity Matrix. Since all structures should be in 2D, then the 
% Connectivity should be of size nX2 where n is the number of rows in the X
% 
% 3). Material Properites and Characteritics in a cell array (Mat)
% Include L (length of the strut), A (cross-sectional area), and 
% E (Young's Modulus) in that given order.
% Ex: MatProp = {L, A, E};
%
% 4/5). If there are varying cross-sections (Aten) or varying E's (Eten) 
% then input a column vector with the apporiate respective values to the 
% struts in there predetermined order. If the cross-sectional areas (A) or 
% the Modulus of Elasticty (E) for each element do not vary, input an empty 
% doulbe array ([])
% 
% 6). The Applied External Force Vector (Fa) to be put into a column 
% vector format. The length of F should be the dimension of the problem 
% statement times the length of the Node position vector X
%
% 7). Constrainted nodes (Constr) for the problem statement
%
% 8). Scaling Factor (s) for the displacement plot
%
%% Outputs: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1). The Resultant Force Vector Fr
% 2).Displacment Vector u
% These are listed out by their enteries
% 3). Two plots, one shows the displacements and the other showing the
% stresses
%
%% Units
% [F] = N
% [K] = N/m
% [L] = mm
% [u] = mm
% [E] = MPa
% [A] = mm^2
% [s] = unitless
% [X] = unitless

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function FEASolverTruss(X, CEM, Mat, Aten, Eten, Fa, Constr, s)
L = Mat{1};
x = L * X; % this x vector has the weights of the struts
dim = size(x, 2); % dimesion is assigned
num_nodes = size(x,1); % number of nodes
num_elements = size(CEM,1); % number of elements

% Checking for varying material properties
if isempty(Aten)
    A = Mat{2} * ones(num_elements, 1);
else
    A = Mat{2} * Aten;
end

if isempty(Eten)
    E = Mat{3} * ones(num_elements,1);
else
    E = Mat{3} * Eten;
end

% Global K Matrix
K = zeros(dim*num_nodes, dim*num_nodes);

for i=1:num_elements
% Organization of elements %
   [~, x_ele, idx_dof_ele] = ElementMap(CEM, x, i);

    % Material Properties and Characteristics of an element
    E_ele = E(i);
    A_ele = A(i);

   Ke = LocalSM(x_ele, E_ele, A_ele);
   Te = LocalTM(x_ele);

% Assembling of the Global K (stiffness) Matrix
    K(idx_dof_ele, idx_dof_ele) = ...
        K(idx_dof_ele, idx_dof_ele) + Te * Ke * Te';
end


% Creates an array of unconstrained nodes
free_dof = true(dim*num_nodes, 1);
free_dof(Constr) = false;

% Global F vector
F = zeros(dim*num_nodes, 1); % Global Force Vector
F = F + Fa;

% Reducing Matrices and Vectors
ReduceK = K(free_dof, free_dof);
ReduceF = F(free_dof);
Reduceu = ReduceK\ReduceF;

% Solutions %
u = zeros(dim*num_nodes,1);
u(free_dof) = Reduceu; % mm
% Resultant Force Vector (Fr)
Fr = K * u; % N

% Final Solutions with Rounding %
RoundFr = round(Fr, 3); % N
Roundu = round(u, 3); % mm

% Displaying Solutions
for i=1:length(Roundu)

disp(['Resultant Force Vector (Fr' sprintf('%d', i) ') : ' ...
    sprintf('%3e',RoundFr(i)) ' N' '  Displacement Vector (u'  ...
    sprintf('%d', i) ') : ' sprintf('%e',Roundu(i)) ' mm'])
end

%% Visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Position vector of elements reshaped in the same dimesion as u
% scaling factor of is applied to the u to create the scaled new positions
% of the struts as the column vector r

R = reshape(x', size(u)); % original positions
r = R + s*u; % Scaled displaced positions

%%%%%%%%%%%%%%%%%%%%%%%% Displacements Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title and Labels
figure;
hold on;
axis equal;
xlabel('x (mm)');
ylabel('y (mm)');
title(['Original and Displaced Positions (scaling factor of ' ...
    sprintf('%d', s) ')']);

% Plottig
for i=1:num_elements
[~, ~, idx_dof_ele] = ElementMap(CEM, x, i);
    R_i = R(idx_dof_ele);
    % plot of original position
    P1 = plot(R_i(1:2:end), R_i(2:2:end), 'k-', 'Marker','o', ...
        LineWidth=1);
    % displaced position of the strut
    r_i = r(idx_dof_ele);
    P2 = plot(r_i(1:2:end), r_i(2:2:end), 'k--', LineWidth=1.2);
end

% Applied External Force Vector plotted (Fa)
for i = 1:dim:num_nodes*dim
    if Fa(i) ~= 0 || Fa(i+1) ~= 0
        P3 = quiver(R(i), R(i+1), Fa(i)/max(abs(Fa))*L/2, ...
            Fa(i+1)/max(abs(Fa))*L/2, ...
               'r', 'LineWidth', 1.5, 'MaxHeadSize', 2);
    end
end
% creating a legend
h = [P1(1) P2(1), P3];
legend(h,'Original Position','Displaced Position', 'Applied Force Vector');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stress Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating Stresses
[Sigma] = Stress(u, CEM, x, A, Mat);

% Title and Labels
figure;
axis equal;
hold on;
xlabel('x (mm)');
ylabel('y (mm)');
title('Stresses in Struts')
% Define the colormap and color limits
clim([min(Sigma) max(Sigma)]);
colormap(jet);
% Create the colorbar
cb = colorbar;
cb.Label.String = '\sigma in [MPa]';
% Set the ticks and labels on the colorbar
sigma_sort = unique(Sigma,'sorted');
cb.Ticks = sigma_sort;
% Mappings of the stress for the struts
cmap = jet;
n = size(jet,1);
sigma_map = (Sigma - min(Sigma))/(max(Sigma)-min(Sigma)) * (n-1) + 1;

% Plotting
for i=1:num_elements
    [~,x_ele,~] = ElementMap(CEM, x, i);
    % Plot with color corresponding to stress
    plot(x_ele(:, 1), x_ele(:, 2), 'LineWidth', 2, 'Color', ...
         cmap(floor(sigma_map(i)), :),'Marker','o'); 
end
hold off;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These are functions found within the FEASolverTruss function

%%%%%%%%%%%%%%% Elemenet Mappings 2D (Organizations) %%%%%%%%%%%%%%%%%%%%%%
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
% Ke = EeAe/Le * [1, -1; -1, 1]
% Stiffness matrix without material properties (B matrix)
b = [1,-1;-1,1];
   % These calculations are done the in Master (Local) Domain
    L_ele = norm(x_ele(2,:)'-x_ele(1,:)');
    % Local Stiffness Matrix
    Ke = E_ele * A_ele/L_ele * b;
end
%%%%%%%%%%%%%%%%%%%%% Local Transformation Matrix %%%%%%%%%%%%%%%%%%%%%%%%%
function Te = LocalTM(x_ele)
% Angle between struts
phi_ele = atan((x_ele(2,2)-x_ele(1,2))/(x_ele(2,1)-x_ele(1,1)));
% Transforms from Master to Global Domain
Te = [cos(phi_ele) 0 ;
      sin(phi_ele) 0 ;
      0 cos(phi_ele) ;
      0 sin(phi_ele)];
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