/// Padding utilised in [FiltFilt::filtfilt].
// WARN: Related/Duplicate: [super::Pad].
#[derive(Copy, Clone, Default)]
pub enum FiltFiltPadType {
    /// Odd extensions
    #[default]
    Odd,
    /// Even extensions
    Even,
    /// Constant extensions
    Const,
}
