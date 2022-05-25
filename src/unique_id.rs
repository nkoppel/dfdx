///! A simple implementation of a UID.

/// An id used in [Gradients] and Tensors to associate gradients with Tensors.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct UniqueId(pub(crate) usize);

/// Generate a [UniqueId].
pub(crate) fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

impl std::ops::Deref for UniqueId {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Something that has a [UniqueId]
pub trait HasUniqueId {
    fn id(&self) -> &UniqueId;
}