import unittest

from gce.config import (
    DEFAULT_GCE_T_MAX,
    DEFAULT_PARAMS,
    DEFAULT_VIEW_T_MAX,
    GCE_T_MAX_MAX,
    coerce_solver_params,
)
from server import DEFAULT_STAR_COUNT, app


class PriorityFixesTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _create_galaxy(self, **overrides):
        response = self.client.post('/api/galaxy', json={
            'n_stars': 200,
            't_max': DEFAULT_GCE_T_MAX,
            'view_t_max': DEFAULT_VIEW_T_MAX,
            **overrides,
        })
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_defaults_round_trip_without_oom(self):
        defaults = self.client.get('/api/defaults').get_json()
        self.assertEqual(defaults['n_stars'], DEFAULT_STAR_COUNT)
        self.assertEqual(defaults['t_max'], DEFAULT_PARAMS['t_max'])
        self.assertEqual(defaults['view_t_max'], DEFAULT_VIEW_T_MAX)

        response = self.client.post('/api/galaxy', json={
            'n_stars': 10,
            't_max': defaults['t_max'],
            'view_t_max': defaults['view_t_max'],
        })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['t_max'], DEFAULT_GCE_T_MAX)
        self.assertEqual(payload['view_t_max'], DEFAULT_VIEW_T_MAX)
        self.assertTrue(payload['cache_id'])

    def test_solver_rejects_unsafe_tmax(self):
        with self.assertRaises(ValueError):
            coerce_solver_params({'t_max': GCE_T_MAX_MAX + 1.0})

    def test_lv_frac_override_changes_rocky_planet(self):
        payload = self._create_galaxy()
        cache_id = payload['cache_id']

        for star_id in range(50):
            base = self.client.get(f'/api/star/{star_id}?t=13.8&cache_id={cache_id}').get_json()
            alt = self.client.get(f'/api/star/{star_id}?t=13.8&lv_frac=0.1&cache_id={cache_id}').get_json()
            if not isinstance(base, dict) or not isinstance(alt, dict):
                continue

            for idx, planet in enumerate(base.get('planets', [])):
                if planet.get('type') not in ('rocky', 'hot_rocky'):
                    continue
                if planet.get('status') != 'active':
                    continue

                base_lv = planet.get('late_veneer', {}).get('lv_mass_frac')
                alt_lv = alt['planets'][idx].get('late_veneer', {}).get('lv_mass_frac')
                self.assertNotEqual(base_lv, alt_lv)
                self.assertAlmostEqual(alt_lv, 0.1, places=3)
                return

        self.fail('Could not find an active rocky planet to validate lv_frac override')

    def test_cache_id_isolates_multiple_galaxies(self):
        early = self._create_galaxy(t_max=5.0)
        late = self._create_galaxy(t_max=20.0, sfr_efficiency=0.2)

        early_id = early['cache_id']
        late_id = late['cache_id']
        self.assertNotEqual(early_id, late_id)

        for star_id in range(30):
            star_early = self.client.get(
                f'/api/star/{star_id}?t=4.5&cache_id={early_id}'
            ).get_json()
            star_late = self.client.get(
                f'/api/star/{star_id}?t=4.5&cache_id={late_id}'
            ).get_json()
            if not isinstance(star_early, dict) or not isinstance(star_late, dict):
                continue

            signature_early = (
                star_early.get('birth_time'),
                star_early.get('birth_z'),
                star_early.get('metallicity'),
                star_early.get('star_mass'),
            )
            signature_late = (
                star_late.get('birth_time'),
                star_late.get('birth_z'),
                star_late.get('metallicity'),
                star_late.get('star_mass'),
            )
            if signature_early == signature_late:
                continue

            latest = self.client.get(f'/api/star/{star_id}?t=4.5').get_json()
            latest_signature = (
                latest.get('birth_time'),
                latest.get('birth_z'),
                latest.get('metallicity'),
                latest.get('star_mass'),
            )
            self.assertEqual(latest_signature, signature_late)
            return

        self.fail('Could not find a star whose cached payload differs across galaxy runs')


if __name__ == '__main__':
    unittest.main()
