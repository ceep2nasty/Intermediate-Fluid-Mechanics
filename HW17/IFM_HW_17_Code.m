function IFM_HW_17_Code()
    % problem1(2,200);
    % problem1(3,200);
    % problem1(4,200);
    % problem1(10,200);
    problem2();    % N=51 at t = [0, 0.01, 0.1, 1, 10]
    make_vortex_movie(51, 10, 60, 0, 'vortices_N51_fast.mp4');

end

% 7.8
function problem1(N, tEnd)
    [x0, y0] = initial_positions(N);
    z0 = [x0; y0];

    f = @(t,z) rhs_point_vortices(t, z, 1.0);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-10);  

    % integrate and keep all time history for simple plotting
    sol = ode45(f, [0, tEnd], z0, opts);

    %trajectory of the vortex that started at (-1,0) 
    tt = linspace(0, tEnd, 2000);
    ZZ = deval(sol, tt);
    x = ZZ(1:N, :);           
    y = ZZ(N+1:2*N, :);

    figure('Color','w'); 
    plot(x(1,:), y(1,:), 'LineWidth', 1.4);
    title(sprintf('Trajectory of vortex starting at (-1,0), N = %d', N));
    xlabel('x'); ylabel('y'); axis equal; grid on;
end

%7.9
function problem2()
    N = 51;
    snaps = [0, 0.01, 0.1, 1, 10];
    [x0, y0] = initial_positions(N);
    z0 = [x0; y0];

    f = @(t,z) rhs_point_vortices(t, z, 1.0);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

    sol = ode45(f, [0, max(snaps)], z0, opts);

    for tSnap = snaps
        z = deval(sol, tSnap);
        x = z(1:N); 
        y = z(N+1:2*N);

        figure('Color','w');
        scatter(x, y, 18, 'filled');
        title(sprintf('N%d vortices at t = %g', N, tSnap));
        xlabel('x'); ylabel('y'); axis equal; grid on;
    end
end

function make_vortex_movie(N, tEnd, fps, tailSec, outFile, varargin)

    p = inputParser;
    p.addParameter('showWindow', [0 tEnd], @(v)isnumeric(v)&&numel(v)==2);
    p.addParameter('axisMargin', 0.1, @isscalar);
    p.addParameter('gamma', 1.0, @isscalar);
    p.parse(varargin{:});
    tShow = p.Results.showWindow;
    mar   = p.Results.axisMargin;
    Gamma = p.Results.gamma;

    % ---- Initial state
    x0 = linspace(-1, 1, N).'; 
    y0 = zeros(N,1);
    z0 = [x0; y0];

    % ---- Integrate 
    f    = @(t,z) rhs_point_vortices(t, z, Gamma);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
    sol  = ode45(f, [0 tEnd], z0, opts);

    % ---- Choose frame times
    if tShow(1) < 0 || tShow(2) > tEnd
        error('showWindow must lie within [0, tEnd].');
    end
    durationSec = tShow(2) - tShow(1);
    nFrames = max(2, round(durationSec*fps));
    tFrames = linspace(tShow(1), tShow(2), nFrames);

    % ---- Evaluate solution on the frame grid
    Z = deval(sol, tFrames);         
    X = Z(1:N,   :);
    Y = Z(N+1:end, :);

    % ---- Pre-compute axis limits with a small margin
    xmin = min(X(:)); xmax = max(X(:));
    ymin = min(Y(:)); ymax = max(Y(:));
    dx = xmax - xmin; dy = ymax - ymin;
    if dx == 0, dx = 1; end
    if dy == 0, dy = 1; end
    xlims = [xmin - mar*dx, xmax + mar*dx];
    ylims = [ymin - mar*dy, ymax + mar*dy];

    % ---- Tail length in frames
    tailFrames = max(1, round(tailSec*fps));

    % ---- Set up figure & graphics objects
    fig = figure('Color','w','Position',[100 100 720 720]);
    ax = axes('Parent',fig); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
    xlim(ax, xlims); ylim(ax, ylims);

    xlabel(ax,'x'); ylabel(ax,'y');

    % Tails: one line per vortex (kept short for speed)
    tailColor = [0.3 0.3 0.3];
    tailWidth = 0.75;
    hTail = gobjects(N,1);
    for i = 1:N
        hTail(i) = plot(ax, X(i,1), Y(i,1), '-', ...
            'Color', tailColor, 'LineWidth', tailWidth);
    end

    % Points
    hPts = scatter(ax, X(:,1), Y(:,1), 28, 'filled');

    % ---- Video writer
    v = VideoWriter(outFile, 'MPEG-4');
    v.FrameRate = fps;
    open(v);

    % ---- Render loop
    for k = 1:nFrames
        k0 = max(1, k - tailFrames + 1);
        % Update tails and points
        for i = 1:N
            set(hTail(i), 'XData', X(i,k0:k), 'YData', Y(i,k0:k));
        end
        set(hPts, 'XData', X(:,k), 'YData', Y(:,k));

        drawnow;
        writeVideo(v, getframe(fig));
    end
    close(v);
    disp(['Saved movie: ' outFile]);
end

%Local Heleprs

function [x0, y0] = initial_positions(N)
    x0 = linspace(-1, 1, N).';
    y0 = zeros(N, 1);
end

function dz = rhs_point_vortices(~, z, Gamma)

    N = numel(z)/2;
    x = z(1:N);
    y = z(N+1:end);

    dx = x - x.';         
    dy = y - y.';         
    r2 = dx.^2 + dy.^2;

    r2(1:N+1:end) = Inf;
    invr2 = 1 ./ r2;

    coef = Gamma/(2*pi);
    u_mat = -coef * (dy .* invr2);
    v_mat =  coef * (dx .* invr2);

    xdot = sum(u_mat, 2);
    ydot = sum(v_mat, 2);
    dz = [xdot; ydot];
end
