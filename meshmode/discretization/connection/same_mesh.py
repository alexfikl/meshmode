__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
from meshmode.transform_metadata import DiscretizationElementAxisTag

from arraycontext import ArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.connection.direct import DirectDiscretizationConnection


# {{{ same-mesh constructor

def make_same_mesh_connection(
        actx: ArrayContext,
        to_discr: Discretization,
        from_discr: Discretization,
        strict: bool = True) -> DirectDiscretizationConnection:
    """
    :arg strict: if *True*, the two discretizations are expected to have the
        same underyling mesh (in the sense of object equivalence). Otherwise,
        just the structure of the group layout is checked (i.e. same types
        of element groups, same number of elements, etc.).

    :returns: a
        :class:`~meshmode.discretization.connection.DirectDiscretizationConnection`
        between the two discretizations.
    """

    from meshmode.discretization.connection.direct import (
            InterpolationBatch,
            DiscretizationConnectionElementGroup,
            IdentityDiscretizationConnection)

    if strict:
        if from_discr.mesh is not to_discr.mesh:
            raise ValueError(
                    "'from_discr' and 'to_discr' must be based on the same mesh")

    if from_discr is to_discr:
        return IdentityDiscretizationConnection(from_discr)

    if not strict:
        if (from_discr.ambient_dim != to_discr.ambient_dim
                or from_discr.dim != to_discr.dim):
            raise ValueError("discretizations have different dimensions")

        if len(from_discr.groups) != len(to_discr.groups):
            raise ValueError(
                    "discretizations do not have same number of groups; got "
                    f"{len(from_discr.groups)} 'from_discr' and "
                    f"{len(to_discr.groups)} 'to_discr' groups")

    groups = []
    for igrp, (fgrp, tgrp) in enumerate(zip(from_discr.groups, to_discr.groups)):
        if not strict:
            if (type(fgrp.mesh_el_group)
                    is not type(tgrp.mesh_el_group)):  # noqa: E721
                raise TypeError(
                        "groups must have the same underlying MeshElementGroup; "
                        f"got '{type(fgrp.mesh_el_group).__name__}' in 'from_discr'"
                        f"and '{type(tgrp.mesh_el_group).__name__}' in 'to_discr'")

            if fgrp.nelements != tgrp.nelements:
                raise ValueError(
                        f"group '{igrp}' has different number of elements; "
                        f"got {fgrp.nelements} 'from_discr' "
                        f"and {tgrp.nelements} 'to_discr' elements")

        from arraycontext.metadata import NameHint
        all_elements = actx.tag(NameHint(f"all_el_ind_grp{igrp}"),
                    actx.tag_axis(0,
                        DiscretizationElementAxisTag(),
                        actx.from_numpy(
                            np.arange(
                                fgrp.nelements,
                                dtype=np.intp))))
        all_elements = actx.freeze(all_elements)

        ibatch = InterpolationBatch(
                from_group_index=igrp,
                from_element_indices=all_elements,
                to_element_indices=all_elements,
                result_unit_nodes=tgrp.unit_nodes,
                to_element_face=None)

        groups.append(DiscretizationConnectionElementGroup([ibatch]))

    return DirectDiscretizationConnection(
            from_discr, to_discr, groups,
            is_surjective=True)

# }}}

# vim: foldmethod=marker
