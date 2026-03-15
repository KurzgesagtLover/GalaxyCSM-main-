import unittest

from server import app


class ModesApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _run_galaxy(self, **payload):
        response = self.client.post('/api/galaxy', json={'n_stars': 100, 't_max': 20.0, **payload})
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_parameter_modes_change_outputs(self):
        base = self._run_galaxy()
        s_process = self._run_galaxy(yield_s_multiplier=5.0, yield_r_multiplier=1.0)
        r_process = self._run_galaxy(yield_s_multiplier=1.0, yield_r_multiplier=0.1)
        galaxy = self._run_galaxy(sfr_efficiency=0.30, outflow_eta=2.0)
        ia_agb = self._run_galaxy(yield_ia_multiplier=5.0, agb_frequency_multiplier=5.0)

        gce1 = base['gce']
        gce2 = s_process['gce']
        gce3 = r_process['gce']
        gce4 = galaxy['gce']
        gce5 = ia_agb['gce']

        ba1 = gce1['mass_fractions']['Ba'][-1][-1]
        ba2 = gce2['mass_fractions']['Ba'][-1][-1]
        eu1 = gce1['mass_fractions']['Eu'][-1][-1]
        eu3 = gce3['mass_fractions']['Eu'][-1][-1]
        mass_star_1 = gce1['stellar_mass'][-1][-1]
        mass_star_4 = gce4['stellar_mass'][-1][-1]
        fe1 = gce1['mass_fractions']['Fe'][-1][-1]
        fe5 = gce5['mass_fractions']['Fe'][-1][-1]
        ba5 = gce5['mass_fractions']['Ba'][-1][-1]

        self.assertGreater(ba2, ba1)
        self.assertLess(eu3, eu1)
        self.assertNotEqual(mass_star_4, mass_star_1)
        self.assertGreater(fe5, fe1)
        self.assertGreater(ba5, ba1)


if __name__ == '__main__':
    unittest.main()
