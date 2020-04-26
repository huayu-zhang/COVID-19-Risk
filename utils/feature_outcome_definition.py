import json
import pandas as pd

with open('project_params.json', 'r') as f:
    params = json.load(f)


class AdmissionContext:

    def __init__(self):
        self.column_description = pd.read_csv(params['COLUMN_DESCRIPTION_FILE'], sep='\t', index_col=0)

        self.feature_keys = [key for key in self.column_description.admission_context.unique() if 'feature' in str(key)]
        self.feature_sets = {
            key: list(self.column_description.column_name[self.column_description.admission_context == key])
            for key in self.feature_keys}

        self.feature_dict = {
            'basic': self.feature_sets['feature_basic'],
            'basic_condi': self.feature_sets['feature_basic'] + self.feature_sets['feature_pre-existing-condition'],
            'basic_symp': self.feature_sets['feature_basic'] + self.feature_sets['feature_symptom'],
            'basic_lab': self.feature_sets['feature_basic'] + self.feature_sets['feature_laboratory_test'],
            'basic_condi_symp': self.feature_sets['feature_basic'] +
            self.feature_sets['feature_pre-existing-condition'] +
            self.feature_sets['feature_symptom'],
            'basic_condi_lab': self.feature_sets['feature_basic'] +
            self.feature_sets['feature_pre-existing-condition'] +
            self.feature_sets['feature_laboratory_test'],
            'basic_symp_lab': self.feature_sets['feature_basic'] +
            self.feature_sets['feature_symptom'] +
            self.feature_sets['feature_laboratory_test'],
            'basic_condi_symp_lab': self.feature_sets['feature_basic'] +
            self.feature_sets['feature_symptom'] +
            self.feature_sets['feature_pre-existing-condition'] +
            self.feature_sets['feature_laboratory_test'],
            'all': self.feature_sets['feature_basic'] +
            self.feature_sets['feature_pre-existing-condition'] +
            self.feature_sets['feature_symptom'] +
            self.feature_sets['feature_laboratory_test'] +
            self.feature_sets['feature_assessment'] +
            self.feature_sets['feature_extra']
        }

        self.outcome_keys = [key for key in self.column_description.admission_context.unique() if 'outcome' in str(key)]
        self.outcome_sets = {
            key: list(self.column_description.column_name[self.column_description.admission_context == key])
            for key in self.outcome_keys}
        self.outcome_dict = {key: [key] for key in self.outcome_sets['outcome_1']}


class FollowUpContext:

    def __init__(self):
        self.column_description = pd.read_csv(params['COLUMN_DESCRIPTION_FILE'], sep='\t', index_col=0)

        self.feature_keys = [key for key in self.column_description.followup_context.unique() if 'feature' in str(key)]
        self.feature_sets = {
            key: list(self.column_description.column_name[self.column_description.followup_context == key])
            for key in self.feature_keys}

        self.feature_dict = {
            'basic': self.feature_sets['feature_basic'],
            'basic_condi': self.feature_sets['feature_basic'] + self.feature_sets['feature_pre-existing-condition'],
            'basic_symp': self.feature_sets['feature_basic'] + self.feature_sets['feature_symptom'],
            'basic_lab': self.feature_sets['feature_basic'] + self.feature_sets['feature_laboratory_test'],
            'basic_antibody': self.feature_sets['feature_basic'] + self.feature_sets['feature_antibody_titre'],
            'basic_course': self.feature_sets['feature_basic'] + self.feature_sets['feature_course_of_disease'],
            'basic_assessment': self.feature_sets['feature_basic'] +
            self.feature_sets['feature_assessment_admission'] +
            self.feature_sets['feature_assessment_midcourse'] +
            self.feature_sets['feature_assessment_end'],
            'all': [index for index in self.column_description.index
                    if not pd.isna(self.column_description.followup_context[index])
                    if 'feature' in self.column_description.followup_context[index]]
        }

        self.outcome_keys = [key for key in self.column_description.followup_context.unique() if 'outcome' in str(key)]
        self.outcome_sets = {
            key: list(self.column_description.column_name[self.column_description.followup_context == key])
            for key in self.outcome_keys}
        self.outcome_dict = {key: [key] for key in self.outcome_sets['outcome_1']}

