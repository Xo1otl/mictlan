# syuron

pythonだけフォルダ名とパッケージ名の一致が必須なのでsyuronという名前になっている


## prompt

function [Hz,Ex,Ey,Ez] = postprocess (lambda,neff,Hx,Hy,dx,dy,varargin);
% This function takes the two computed transverse magnetic
% fields (Hx and Hy) of an optical waveguide structure and
% solves for the remaining 4 vield components:  Hz, Ex, Ey,
% and Ez.
%
% USAGE:
% 
% [Hz,Ex,Ey,Ez] = postprocess(lambda, neff, Hx, Hy, dx, dy, ...
%                     eps, boundary);
% [Hz,Ex,Ey,Ez] = postprocess(lambda, neff, Hx, Hy, dx, dy, ...
%                     epsxx, epsyy, epszz, boundary);
% [Hz,Ex,Ey,Ez] = postprocess(lambda, neff, Hx, Hy, dx, dy, ...
%                     epsxx, epsxy, epsyx, epsyy, epszz, boundary);
% 
% INPUT:
% 
% lambda - optical wavelength at which mode was calculated
% neff - the calculated effective index of the optial mode
% Hx, Hy - the calculated transverse magnetic fields of the mode
% dx - horizontal grid spacing (vector or scalar)
% dy - vertical grid spacing (vector or scalar)
% eps - index mesh (isotropic materials)  OR:
% epsxx, epsxy, epsyx, epsyy, epszz - index mesh (anisotropic)
% boundary - 4 letter string specifying boundary conditions to be
% applied at the edges of the computation window.  
%   boundary(1) = North boundary condition
%   boundary(2) = South boundary condition
%   boundary(3) = East boundary condition
%   boundary(4) = West boundary condition
% The following boundary conditions are supported: 
%   'A' - Hx is antisymmetric, Hy is symmetric.
%   'S' - Hx is symmetric and, Hy is antisymmetric.
%   '0' - Hx and Hy are zero immediately outside of the
%         boundary. 
% 
% OUTPUT:
% 
% Hz - calculated longitudinal magnetic field.  This output will 
%   have the same dimensions as Hx and Hy.
% Ex, Ey, Ez - calculated electric field.  These field components 
%   are computed at the center of each element instead of on the
%   edges or vertices.
%
% NOTES:
%
% 1) This routine is meant to be used in conjunction with
% wgmodes.m, the vector eigenmode solver.  Please consult the
% help file for wgmodes.m for more information.
%
% 2) The boundary conditions and waveguide specifications
% (given in dx, dy, eps, and boundary) should be the same as
% what was used in wgmodes.m to compute the mode.
%
% 3) The magnetic field components (Hx, Hy, and Hz) are
% calculated at the edges of each cell, whereas the electric
% field components are computed at the center of each cell.
% Therefore if size(eps) = [n,m], then the magnetic fields
% will have a size of [n+1,m+1] while the computed electric
% fields will have a size of [n,m].
%
% 4) Even though wgmodes.m will optionally calculate more than
% one mode at a time, this postprocessing routine must be
% invoked separately for each computed mode.
%
% AUTHORS:  Thomas E. Murphy (tem@umd.edu)
if (nargin == 12)
  epsxx = varargin{1};
  epsxy = varargin{2};
  epsyx = varargin{3};
  epsyy = varargin{4};
  epszz = varargin{5};
  boundary = varargin{6};
elseif (nargin == 10)
  epsxx = varargin{1};
  epsxy = zeros(size(epsxx));
  epsyx = zeros(size(epsxx));
  epsyy = varargin{2};
  epszz = varargin{3};
  boundary = varargin{4};
elseif (nargin == 8)
  epsxx = varargin{1};
  epsxy = zeros(size(epsxx));
  epsyx = zeros(size(epsxx));
  epsyy = epsxx;
  epszz = epsxx;
  boundary = varargin{2};
else
  error('Incorrect number of input arguments.\n');
end
[nx,ny] = size(epsxx);
nx = nx + 1;
ny = ny + 1;
% now we pad eps on all sides by one grid point
epsxx = [epsxx(:,1),epsxx,epsxx(:,ny-1)];
epsxx = [epsxx(1,1:ny+1);epsxx;epsxx(nx-1,1:ny+1)];
epsyy = [epsyy(:,1),epsyy,epsyy(:,ny-1)];
epsyy = [epsyy(1,1:ny+1);epsyy;epsyy(nx-1,1:ny+1)];
epsxy = [epsxy(:,1),epsxy,epsxy(:,ny-1)];
epsxy = [epsxy(1,1:ny+1);epsxy;epsxy(nx-1,1:ny+1)];
epsyx = [epsyx(:,1),epsyx,epsyx(:,ny-1)];
epsyx = [epsyx(1,1:ny+1);epsyx;epsyx(nx-1,1:ny+1)];
epszz = [epszz(:,1),epszz,epszz(:,ny-1)];
epszz = [epszz(1,1:ny+1);epszz;epszz(nx-1,1:ny+1)];
k = 2*pi/lambda;  % free-space wavevector
b = neff*k;       % propagation constant (eigenvalue)
if isscalar(dx)
  dx = dx*ones(nx+1,1);             % uniform grid
else
  dx = dx(:);                       % convert to column vector
  dx = [dx(1);dx;dx(length(dx))];   % pad dx on top and bottom
end
if isscalar(dy)
  dy = dy*ones(1,ny+1);             % uniform grid
else
  dy = dy(:);                       % convert to column vector
  dy = [dy(1);dy;dy(length(dy))]';  % pad dy on top and bottom
end
% distance to neighboring points to north south east and west,
% relative to point under consideration (P), as shown below.
n = ones(1,nx*ny);      n(:) = ones(nx,1)*dy(2:ny+1);
s = ones(1,nx*ny);      s(:) = ones(nx,1)*dy(1:ny);
e = ones(1,nx*ny);      e(:) = dx(2:nx+1)*ones(1,ny);
w = ones(1,nx*ny);      w(:) = dx(1:nx)*ones(1,ny);
% epsilon tensor elements in regions 1,2,3,4, relative to the
% mesh point under consideration (P), as shown below.
%
%                 NW------N------NE
%                 |       |       |
%                 |   1   n   4   |
%                 |       |       |
%                 W---w---P---e---E
%                 |       |       |
%                 |   2   s   3   |
%                 |       |       |
%                 SW------S------SE
exx1 = ones(1,nx*ny);   exx1(:) = epsxx(1:nx,2:ny+1);
exx2 = ones(1,nx*ny);   exx2(:) = epsxx(1:nx,1:ny);
exx3 = ones(1,nx*ny);   exx3(:) = epsxx(2:nx+1,1:ny);
exx4 = ones(1,nx*ny);   exx4(:) = epsxx(2:nx+1,2:ny+1);
eyy1 = ones(1,nx*ny);   eyy1(:) = epsyy(1:nx,2:ny+1);
eyy2 = ones(1,nx*ny);   eyy2(:) = epsyy(1:nx,1:ny);
eyy3 = ones(1,nx*ny);   eyy3(:) = epsyy(2:nx+1,1:ny);
eyy4 = ones(1,nx*ny);   eyy4(:) = epsyy(2:nx+1,2:ny+1);
exy1 = ones(1,nx*ny);   exy1(:) = epsxy(1:nx,2:ny+1);
exy2 = ones(1,nx*ny);   exy2(:) = epsxy(1:nx,1:ny);
exy3 = ones(1,nx*ny);   exy3(:) = epsxy(2:nx+1,1:ny);
exy4 = ones(1,nx*ny);   exy4(:) = epsxy(2:nx+1,2:ny+1);
eyx1 = ones(1,nx*ny);   eyx1(:) = epsyx(1:nx,2:ny+1);
eyx2 = ones(1,nx*ny);   eyx2(:) = epsyx(1:nx,1:ny);
eyx3 = ones(1,nx*ny);   eyx3(:) = epsyx(2:nx+1,1:ny);
eyx4 = ones(1,nx*ny);   eyx4(:) = epsyx(2:nx+1,2:ny+1);
ezz1 = ones(1,nx*ny);   ezz1(:) = epszz(1:nx,2:ny+1);
ezz2 = ones(1,nx*ny);   ezz2(:) = epszz(1:nx,1:ny);
ezz3 = ones(1,nx*ny);   ezz3(:) = epszz(2:nx+1,1:ny);
ezz4 = ones(1,nx*ny);   ezz4(:) = epszz(2:nx+1,2:ny+1);
bzxne = 
bzxse = 
bzxnw = 
bzxsw = 
bzxn = 
bzxs =
bzxe = 
bzxw = 
bzxp = 
bzyne = 
bzyse = 
bzynw = 
bzysw = 
bzyn = 
bzys = 
bzye = 
bzyw = 
bzyp = 
ii = zeros(nx,ny);
ii(:) = (1:nx*ny); 
% NORTH boundary
ib = zeros(nx,1);  ib(:) = ii(1:nx,ny);
switch (boundary(1))
  case 'S',   sign = +1;
  case 'A',   sign = -1;
  case '0',   sign = 0;
  otherwise,  
    error('Unrecognized north boundary condition: %s.\n', boundary(1));
end
bzxs(ib)  = bzxs(ib)  + sign*bzxn(ib);
bzxse(ib) = bzxse(ib) + sign*bzxne(ib);
bzxsw(ib) = bzxsw(ib) + sign*bzxnw(ib);
bzys(ib)  = bzys(ib)  - sign*bzyn(ib);
bzyse(ib) = bzyse(ib) - sign*bzyne(ib);
bzysw(ib) = bzysw(ib) - sign*bzynw(ib);
% SOUTH boundary
ib = zeros(nx,1);  ib(:) = ii(1:nx,1);
switch (boundary(2))
  case 'S',   sign = +1;
  case 'A',   sign = -1;
  case '0',   sign = 0;
  otherwise,  
    error('Unrecognized south boundary condition: %s.\n', boundary(2));
end
bzxn(ib)  = bzxn(ib)  + sign*bzxs(ib);
bzxne(ib) = bzxne(ib) + sign*bzxse(ib);
bzxnw(ib) = bzxnw(ib) + sign*bzxsw(ib);
bzyn(ib)  = bzyn(ib)  - sign*bzys(ib);
bzyne(ib) = bzyne(ib) - sign*bzyse(ib);
bzynw(ib) = bzynw(ib) - sign*bzysw(ib);
% EAST boundary
ib = zeros(1,ny);  ib(:) = ii(nx,1:ny);
switch (boundary(3))
  case 'S',   sign = +1;
  case 'A',   sign = -1;
  case '0',   sign = 0;
  otherwise,  
    error('Unrecognized east boundary condition: %s.\n', boundary(3));
end
bzxw(ib)  = bzxw(ib)  + sign*bzxe(ib);
bzxnw(ib) = bzxnw(ib) + sign*bzxne(ib);
bzxsw(ib) = bzxsw(ib) + sign*bzxse(ib);
bzyw(ib)  = bzyw(ib)  - sign*bzye(ib);
bzynw(ib) = bzynw(ib) - sign*bzyne(ib);
bzysw(ib) = bzysw(ib) - sign*bzyse(ib);
% WEST boundary
ib = zeros(1,ny);  ib(:) = ii(1,1:ny);
switch (boundary(4))
  case 'S',   sign = +1;
  case 'A',   sign = -1;
  case '0',   sign = 0;
  otherwise,  
    error('Unrecognized west boundary condition: %s.\n', boundary(4));
end
bzxe(ib)  = bzxe(ib)  + sign*bzxw(ib);
bzxne(ib) = bzxne(ib) + sign*bzxnw(ib);
bzxse(ib) = bzxse(ib) + sign*bzxsw(ib);
bzye(ib)  = bzye(ib)  - sign*bzyw(ib);
bzyne(ib) = bzyne(ib) - sign*bzynw(ib);
bzyse(ib) = bzyse(ib) - sign*bzysw(ib);
% Assemble sparse matrix
iall = zeros(1,nx*ny);          iall(:) = ii;
is = zeros(1,nx*(ny-1));        is(:) = ii(1:nx,1:(ny-1));
in = zeros(1,nx*(ny-1));        in(:) = ii(1:nx,2:ny);
ie = zeros(1,(nx-1)*ny);        ie(:) = ii(2:nx,1:ny);
iw = zeros(1,(nx-1)*ny);        iw(:) = ii(1:(nx-1),1:ny);
ine = zeros(1,(nx-1)*(ny-1));   ine(:) = ii(2:nx, 2:ny);
ise = zeros(1,(nx-1)*(ny-1));   ise(:) = ii(2:nx, 1:(ny-1));
isw = zeros(1,(nx-1)*(ny-1));   isw(:) = ii(1:(nx-1), 1:(ny-1));
inw = zeros(1,(nx-1)*(ny-1));   inw(:) = ii(1:(nx-1), 2:ny);
Bzx = sparse ([iall,iw,ie,is,in,ine,ise,isw,inw], ...
	[iall,ie,iw,in,is,isw,inw,ine,ise], ...
	[bzxp(iall),bzxe(iw),bzxw(ie),bzxn(is),bzxs(in), ...
     bzxsw(ine),bzxnw(ise),bzxne(isw),bzxse(inw)]);
Bzy = sparse ([iall,iw,ie,is,in,ine,ise,isw,inw], ...
	[iall,ie,iw,in,is,isw,inw,ine,ise], ...
	[bzyp(iall),bzye(iw),bzyw(ie),bzyn(is),bzys(in), ...
     bzysw(ine),bzynw(ise),bzyne(isw),bzyse(inw)]);
B = [Bzx Bzy];
Hz = zeros(size(Hx));
Hz(:) = B*reshape([Hx,Hy],2*nx*ny,1)/j;
nx = nx-1;
ny = ny-1;
exx = epsxx(2:nx+1,2:ny+1);
exy = epsxy(2:nx+1,2:ny+1);
eyx = epsyx(2:nx+1,2:ny+1);
eyy = epsyy(2:nx+1,2:ny+1);
ezz = epszz(2:nx+1,2:ny+1);
edet = (exx.*eyy - exy.*eyx);
h = dx(2:nx+1)*ones(1,ny);
v = ones(nx,1)*dy(2:ny+1);
i1 = ii(1:nx,2:ny+1);
i2 = ii(1:nx,1:ny);
i3 = ii(2:nx+1,1:ny);
i4 = ii(2:nx+1,2:ny+1);
Dx = +neff*(Hy(i1) + Hy(i2) + Hy(i3) + Hy(i4))/4 + ...
     (Hz(i1) + Hz(i4) - Hz(i2) - Hz(i3))./(j*2*k*v);
Dy = -neff*(Hx(i1) + Hx(i2) + Hx(i3) + Hx(i4))/4 - ...
     (Hz(i3) + Hz(i4) - Hz(i1) - Hz(i2))./(j*2*k*h);
Dz = ((Hy(i3) + Hy(i4) - Hy(i1) - Hy(i2))./(2*h) - ...
      (Hx(i1) + Hx(i4) - Hx(i2) - Hx(i3))./(2*v))/(j*k);
Ex = (eyy.*Dx - exy.*Dy)./edet;
Ey = (exx.*Dy - eyx.*Dx)./edet;
Ez = Dz./ezz;

