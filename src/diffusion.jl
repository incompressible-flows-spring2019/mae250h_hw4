# This function takes as input a set of edge data and returns the same
# data with the boundary and ghost values set as needed to enforce Dirichlet
# boundary conditions.
function apply_bc!(u::EdgeData{NX,NY},t::Real) where {NX,NY}
    # on left, right sides, set x edges directly
    u.qx[1,j_e_x]    .= uL.(yedgex.(j_e_x),t)
    u.qx[NX+1,j_e_x] .= uR.(yedgex.(j_e_x),t)

    # on bottom, top sides, set x edges via averaging with ghost edges
    u.qx[i_e_x,1]    .= -u.qx[i_e_x,2]    + 2*uB.(xedgex.(i_e_x),t)
    u.qx[i_e_x,NY+2] .= -u.qx[i_e_x,NY+1] + 2*uT.(xedgex.(i_e_x),t)

    # on left, right sides, set y edges via averaging with ghost edges
    u.qy[1,j_e_y]    .= -u.qy[1,j_e_y]    + 2*vL.(yedgey.(j_e_y),t)
    u.qy[NX+2,j_e_y] .= -u.qy[NX+1,j_e_y] + 2*vR.(yedgey.(j_e_y),t)

    # on bottom, top sides, set y edges directly
    u.qy[i_e_y,1]    .= vB.(xedgey.(i_e_y),t)
    u.qy[i_e_y,NY+1] .= vT.(xedgey.(i_e_y),t)

    return u
end
