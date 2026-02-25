from converter.reader.reader_alignment_core import ReaderAlignmentCoreMixin
from converter.reader.reader_alignment_fps import ReaderAlignmentFpsMixin
from converter.reader.reader_alignment_validation import ReaderAlignmentValidationMixin
from converter.reader.reader_io import ReaderIOMixin
from converter.reader.reader_setup import ReaderSetupMixin
from converter.reader.reader_timestamp import ReaderTimestampMixin


class KuavoRosbagReader(
    ReaderSetupMixin,
    ReaderTimestampMixin,
    ReaderAlignmentCoreMixin,
    ReaderAlignmentFpsMixin,
    ReaderAlignmentValidationMixin,
    ReaderIOMixin,
):
    """Composed rosbag reader from focused mixins."""

    pass
