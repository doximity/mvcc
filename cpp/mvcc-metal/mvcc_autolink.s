// Linker options carried inside the object file (LC_LINKER_OPTION) so that linking libcudart_static.a
// with a bare `-lcudart_static` pulls in the frameworks the Metal shim needs. mvcc_metal.mm references
// the anchor symbol so this member is always loaded from the archive.
	.section	__DATA,__data
	.globl	___mvcc_autolink_anchor
	.p2align	2
___mvcc_autolink_anchor:
	.long	0
	.linker_option "-framework", "Metal"
	.linker_option "-framework", "Foundation"
	.linker_option "-framework", "IOKit"
	.linker_option "-framework", "CoreFoundation"
	.linker_option "-lc++"
	.linker_option "-lobjc"
.subsections_via_symbols
