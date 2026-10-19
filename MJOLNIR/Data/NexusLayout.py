from dataclasses import dataclass, replace
from string import Formatter

@dataclass(frozen=True)
class Field:
    """A dataset in a Nexus file: where it is written, the older paths readers also accept,
    and the units and dtype it is written with (None: no units attribute / keep the data's
    own dtype)"""
    path: str
    aliases: tuple[str, ...] = ()
    units: str | None = None
    dtype: str | None = None

    @property
    def paths(self) -> tuple[str, ...]:
        """Paths in the order readers try them"""
        return (self.path, *self.aliases)

    def format(self, **context) -> 'Field':
        return replace(self, path=_format(self.path, context),
                       aliases=tuple(_format(alias, context) for alias in self.aliases))

@dataclass(frozen=True)
class Group:
    """A group in a Nexus file and its NX_class"""
    path: str
    nx_class: str

    def format(self, **context) -> 'Group':
        return replace(self, path=_format(self.path, context))

def _format(template: str, context: dict) -> str:
    """Fill in the placeholders given in context, leaving the others (e.g. {binning}) in place"""
    return template.format_map(_KeepMissing(context))

class _KeepMissing(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}

class NexusLayout:
    """Where the data of an instrument's Nexus files is stored, shared by its reader and writer.

    Paths are templates: {entry} and {instrument} are filled in from the class attributes of
    the same name, {binning} in calibration paths by calibration_fields(binning). The
    dictionaries hold the templates, so a subclass can change entry or instrument and reuse
    its parent's entries; field(), group() etc. return them filled in. Keys are unique across
    fields, metadata and environment"""

    entry = 'entry'
    instrument: str # Name of the NXinstrument group

    groups: dict[str, Group] = {
        'entry': Group('{entry}', 'NXentry'),
        'instrument': Group('{entry}/{instrument}', 'NXinstrument'),
        'sample': Group('{entry}/sample', 'NXsample'),
        'control': Group('{entry}/control', 'NXmonitor'),
        'data': Group('{entry}/data', 'NXdata'),
        'local_contact': Group('{entry}/local_contact', 'NXuser'),
        'proposal_user': Group('{entry}/proposal_user', 'NXuser'),
        'user': Group('{entry}/user', 'NXuser'),
    }

    # Data used in reduction
    fields: dict[str, Field] = {
        'A3': Field('{entry}/sample/rotation_angle', units='degree', dtype='float32'),
        'A3Offset': Field('{entry}/sample/rotation_angle_zero', units='degree', dtype='float32'),
        'unitCell': Field('{entry}/sample/unit_cell', dtype='float64'),
        'orientationMatrix': Field('{entry}/sample/orientation_matrix', dtype='float32'),
        'planeVector1': Field('{entry}/sample/plane_vector_1', dtype='float32'),
        'planeVector2': Field('{entry}/sample/plane_vector_2', dtype='float32'),
        'planeNormal': Field('{entry}/sample/plane_normal', dtype='float32'),
        'azimuthalAngle': Field('{entry}/sample/azimuthal_angle', dtype='float32'),
        'monitor1': Field('{entry}/control/data', units='counts', dtype='float32'),
        'preset': Field('{entry}/control/preset', dtype='float32'),
        'time': Field('{entry}/control/time', units='seconds', dtype='float32'),
        'absoluteTime': Field('{entry}/control/absolute_time', dtype='float64'),
        'scanCommand': Field('{entry}/scancommand'),
        'scanVariables': Field('{entry}/scanvars'),
    }

    # Data kept for information only (the reader's metadata)
    metadata: dict[str, Field] = {
        'title': Field('{entry}/title'),
        'startTime': Field('{entry}/start_time'),
        'endTime': Field('{entry}/end_time'),
        'comment': Field('{entry}/comment'),
        'proposal': Field('{entry}/proposal_id'),
        'proposalTitle': Field('{entry}/proposal_title'),
        'experimentalIdentifier': Field('{entry}/experiment_identifier'),
        'sampleName': Field('{entry}/sample/name'),
        'localContact': Field('{entry}/local_contact/name'),
        'proposalUser': Field('{entry}/proposal_user/name'),
        'proposalEmail': Field('{entry}/proposal_user/email'),
        'user': Field('{entry}/user/name'),
        'email': Field('{entry}/user/email'),
        'address': Field('{entry}/user/address'),
        'affiliation': Field('{entry}/user/affiliation'),
        'mode': Field('{entry}/control/mode'),
    }

    # Sample environment, keyed by SampleEnvironment field
    environment: dict[str, Field] = {
        'temperature': Field('{entry}/sample/temperature', units='K', dtype='float64'),
        'temperature_setpoint': Field('{entry}/sample/setpoint_temperature', units='K',
                                      dtype='float64'),
        'magnetic_field': Field('{entry}/sample/magnetic_field', units='T', dtype='float64'),
        'magnetic_field_setpoint': Field('{entry}/sample/setpoint_magnetic_field', units='T',
                                         dtype='float64'),
        'electric_field': Field('{entry}/sample/electric_field', dtype='float64'),
        'electric_field_setpoint': Field('{entry}/sample/setpoint_electric_field',
                                         dtype='float64'),
    }

    # Instrument calibration for one binning ({binning}); empty if the files store none
    calibration_group: Group | None = None
    calibration: dict[str, Field] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'instrument'):
            raise TypeError(f'{cls.__name__} must define instrument')
        cls._check_keys()
        cls._check_placeholders()

    @classmethod
    def _check_keys(cls):
        seen = set()
        for table in (cls.fields, cls.metadata, cls.environment):
            repeated = seen & table.keys()
            if repeated:
                raise TypeError(f'{cls.__name__}: keys defined twice: {sorted(repeated)}')
            seen |= table.keys()

    @classmethod
    def _check_placeholders(cls):
        """Catch misspelt placeholders when the class is defined, not when a file is read"""
        known = {'entry', 'instrument'}
        templates = [group.path for group in cls.groups.values()]
        for table in (cls.fields, cls.metadata, cls.environment):
            templates += [path for field in table.values() for path in field.paths]
        calibration = [path for field in cls.calibration.values() for path in field.paths]
        if cls.calibration_group is not None:
            calibration.append(cls.calibration_group.path)
        for template in templates:
            unknown = _placeholders(template) - known
            if unknown:
                raise TypeError(f'{cls.__name__}: unknown placeholder {unknown} in "{template}"')
        for template in calibration:
            unknown = _placeholders(template) - known - {'binning'}
            if unknown:
                raise TypeError(f'{cls.__name__}: unknown placeholder {unknown} in "{template}"')

    @classmethod
    def context(cls) -> dict[str, str]:
        return {'entry': cls.entry, 'instrument': cls.instrument}

    @classmethod
    def field(cls, key: str) -> Field:
        """A field, metadata or environment entry, filled in. KeyError if not in the layout"""
        for table in (cls.fields, cls.metadata, cls.environment):
            if key in table:
                return table[key].format(**cls.context())
        raise KeyError(f'{key} is not in {cls.__name__}')

    @classmethod
    def path(cls, key: str) -> str:
        """Main path of a field: where it is written and read first"""
        return cls.field(key).path

    @classmethod
    def group(cls, key: str) -> Group:
        return cls.groups[key].format(**cls.context())

    @classmethod
    def all_fields(cls) -> dict[str, Field]:
        """Every field, metadata and environment entry, filled in"""
        return {key: field.format(**cls.context())
                for table in (cls.fields, cls.metadata, cls.environment)
                for key, field in table.items()}

    @classmethod
    def all_groups(cls) -> dict[str, Group]:
        """Every group, filled in (the calibration group: calibration_group_for)"""
        return {key: group.format(**cls.context()) for key, group in cls.groups.items()}

    @classmethod
    def calibration_fields(cls, binning: int) -> dict[str, Field]:
        """Calibration datasets for one binning, filled in"""
        return {key: field.format(**cls.context(), binning=binning)
                for key, field in cls.calibration.items()}

    @classmethod
    def calibration_group_for(cls, binning: int) -> Group | None:
        """The calibration group for one binning, filled in; None if the files store none"""
        if cls.calibration_group is None:
            return None
        return cls.calibration_group.format(**cls.context(), binning=binning)

class CameaLayout(NexusLayout):
    """CAMEA (PSI) files, as written by NICOS (2023 and 2024 files checked) and the older
    format read by the old DataFile. Units and dtypes are those of the NICOS files"""

    instrument = 'CAMEA'

    groups = {
        **NexusLayout.groups,
        'analyzer': Group('{entry}/{instrument}/analyzer', 'NXcrystal'),
        'monochromator': Group('{entry}/{instrument}/monochromator', 'NXcrystal'),
        'monochromator_slit': Group('{entry}/{instrument}/monochromator_slit', 'NXslit'),
        'detector': Group('{entry}/{instrument}/detector', 'NXdetector'),
        'segment_1': Group('{entry}/{instrument}/segment_1', 'NXdetector'),
        'segment_8': Group('{entry}/{instrument}/segment_8', 'NXdetector'),
        'monitor_2': Group('{entry}/monitor_2', 'NXmonitor'),
        'proton_beam': Group('{entry}/proton_beam', 'NXmonitor'),
        'temperature_log': Group('{entry}/sample/temperature_log', 'NXlog'),
        'magnetic_field_log': Group('{entry}/sample/magnetic_field_log', 'NXlog'),
    }

    fields = {
        **NexusLayout.fields,
        # Older files store the counts in detector/counts
        'counts': Field('{entry}/{instrument}/detector/data',
                        aliases=('{entry}/{instrument}/detector/counts',), dtype='uint32'),
        'summedCounts': Field('{entry}/{instrument}/detector/summed_counts', units='counts',
                              dtype='int32'),
        'totalCounts': Field('{entry}/{instrument}/detector/total_counts', units='counts',
                             dtype='int32'),
        'detectorSelection': Field('{entry}/{instrument}/detector/detector_selection',
                                   units='selection', dtype='int32'),
        'A4': Field('{entry}/{instrument}/analyzer/polar_angle', units='degree',
                    dtype='float32'),
        'A4Offset': Field('{entry}/{instrument}/analyzer/polar_angle_offset', units='degree',
                          dtype='float32'),
        'A4Raw': Field('{entry}/{instrument}/analyzer/polar_angle_raw', units='degree',
                       dtype='float32'),
        'analyzerSelection': Field('{entry}/{instrument}/analyzer/analyzer_selection',
                                   units='selection', dtype='int32'),
        'analyzerDSpacing': Field('{entry}/{instrument}/analyzer/d_spacing', units='Angstroem',
                                  dtype='float32'),
        'analyzerNominalEnergy': Field('{entry}/{instrument}/analyzer/nominal_energy',
                                       units='mev', dtype='float32'),
        'analyzerType': Field('{entry}/{instrument}/analyzer/type'),
        'Ei': Field('{entry}/{instrument}/monochromator/energy', units='mev', dtype='float32'),
        'monochromatorDSpacing': Field('{entry}/{instrument}/monochromator/d_spacing',
                                       units='Angstroem', dtype='float32'),
        'monochromatorRotationAngle': Field('{entry}/{instrument}/monochromator/rotation_angle',
                                            units='degree', dtype='float32'),
        'monochromatorRotationAngleZero': Field(
            '{entry}/{instrument}/monochromator/rotation_angle_zero', units='degree',
            dtype='float32'),
        'monochromatorType': Field('{entry}/{instrument}/monochromator/type'),
        'singleDetector1': Field('{entry}/{instrument}/segment_1/data', units='counts',
                                 dtype='int32'),
        'singleDetector8': Field('{entry}/{instrument}/segment_8/data', units='counts',
                                 dtype='int32'),
        'monitor2': Field('{entry}/monitor_2/data', units='counts', dtype='int32'),
        'protonBeam': Field('{entry}/proton_beam/data', units='counts', dtype='int32'),
        'sgu': Field('{entry}/sample/sgu', units='degree', dtype='float32'),
        'sguZero': Field('{entry}/sample/sgu_zero', units='degree', dtype='float32'),
        'sgl': Field('{entry}/sample/sgl', units='degree', dtype='float32'),
        'sglZero': Field('{entry}/sample/sgl_zero', units='degree', dtype='float32'),
        'qh': Field('{entry}/sample/qh', units='rlu', dtype='float32'),
        'qk': Field('{entry}/sample/qk', units='rlu', dtype='float32'),
        'ql': Field('{entry}/sample/ql', units='rlu', dtype='float32'),
        # Sample environment logs (2023 files call the temperature T)
        'temperatureLog': Field('{entry}/sample/temperature_log/value',
                                aliases=('{entry}/sample/T_log/value',), dtype='float32'),
        'temperatureLogTime': Field('{entry}/sample/temperature_log/time',
                                    aliases=('{entry}/sample/T_log/time',), dtype='float32'),
        'magneticFieldLog': Field('{entry}/sample/magnetic_field_log/value', dtype='float32'),
        'magneticFieldLogTime': Field('{entry}/sample/magnetic_field_log/time',
                                      dtype='float32'),
    }

    # Where NICOS and older CAMEA files store the sample environment (as the old DataFile)
    environment = {
        **NexusLayout.environment,
        'temperature': Field('{entry}/sample/temperature',
                             aliases=('{entry}/sample/Ts', '{entry}/sample/Ts/value',
                                      '{entry}/sample/se_ts', '{entry}/sample/T'),
                             units='K', dtype='float64'),
        'magnetic_field': Field('{entry}/sample/magnetic_field',
                                aliases=('{entry}/sample/B', '{entry}/sample/B/value'),
                                units='T', dtype='float64'),
    }

    # The files store no units for the calibration
    calibration_group = Group('{entry}/{instrument}/calib{binning}', 'NXcollection')
    calibration = {
        'A4': Field('{entry}/{instrument}/calib{binning}/a4offset', dtype='float32'),
        'amp': Field('{entry}/{instrument}/calib{binning}/amplitude', dtype='float32'),
        'bg': Field('{entry}/{instrument}/calib{binning}/background', dtype='float32'),
        'bound': Field('{entry}/{instrument}/calib{binning}/boundaries', dtype='int32'),
        'Ef': Field('{entry}/{instrument}/calib{binning}/final_energy', dtype='float32'),
        'width': Field('{entry}/{instrument}/calib{binning}/width', dtype='float32'),
    }
