% symbolic 7.6 calculation IFM HW15

clear; clc;


syms x y z real

u = -(x^2) * y * sym(0.5) + x + y + z ;
v = x^2 - y^2 ; 
w = x*y*z + sym(2) * y * z - z;

velocity = [u ; v; w] ;

grad_v = [diff(u, x), diff(u, y), diff(u, z) ;
         diff(v, x), diff(v,y), diff(v, z) ;
         diff(w, x), diff(w, y), diff(w,z)] ;

% show that it is incompressible 

div_v = grad_v(1,1) + grad_v(2,2 )+ grad_v(3,3);

% now do the calculation for w_t dot grad v

omega = curl(velocity, [x, y, z]) ;

% row form of (omega · ∇)v  → (1x3) = (1x3)*(3x3)
tmp = omega.' * grad_v;            % use .' (transpose), not ' (ctranspose)

% force full expansion & simplification entrywise
for k = 1:numel(tmp)
    tmp(k) = simplify(expand(tmp(k)), 'Steps', 500);
end

domega_dt = collect(tmp, [x y z]);  % 1x3 row vector

domega_dt_col = domega_dt.';        % 3x1